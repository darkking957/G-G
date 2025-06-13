# src/got_reasoning/evaluate_plans.py
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

from typing import List, Dict, Any, Tuple, Optional
import json
import asyncio
import logging
from dataclasses import dataclass

from utils import read_prompt, rule_to_string

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """评估结果"""
    score: float
    reasoning: str
    constraints: Dict[str, Any]
    confidence: float
    

class SemanticEvaluator:
    """语义评估器：使用LLM评估路径与问题的语义对齐度"""
    
    def __init__(self, llm_model, prompt_dir: str = "prompts"):
        """
        Args:
            llm_model: LLM模型实例
            prompt_dir: 提示词模板目录
        """
        self.llm_model = llm_model
        self.prompt_dir = prompt_dir
        
        # 加载提示词模板
        self.constraint_prompt = self._load_prompt("constraint_extractor.txt")
        self.evaluator_prompt = self._load_prompt("evaluator_prompt.txt")
        
    def _load_prompt(self, filename: str) -> str:
        """加载提示词模板"""
        try:
            return read_prompt(os.path.join(self.prompt_dir, filename))
        except FileNotFoundError:
            # 如果文件不存在，使用默认模板
            return self._get_default_prompt(filename)
            
    def _get_default_prompt(self, filename: str) -> str:
        """获取默认提示词模板"""
        if filename == "constraint_extractor.txt":
            return """Analyze the question: "{question}". 
Extract key semantic constraints for the answer in JSON format. 
Constraints can include but are not limited to: "entity_type", "gender", "temporal_relation", "spatial_relation", "quantity", "attribute".
If no specific constraint is found, return an empty JSON object {{}}.

Output ONLY valid JSON."""
        
        elif filename == "evaluator_prompt.txt":
            return """**Context:**
- User Question: "{question}"
- Semantic Constraints: {constraints_json}
- Candidate Path: "{path_str}"

**Task:**
Evaluate the semantic alignment of the Candidate Path with the User Question. 
A good path must lead to an answer that satisfies the semantic constraints.

Consider:
1. Does the path capture the core intent of the question?
2. Will following this path likely lead to entities that match the constraints?
3. Is the path logically sound for answering this type of question?

Provide a score from 0.0 to 1.0 and a brief justification.
Output ONLY in JSON format: {{"score": <float>, "reasoning": "<string>"}}"""
        
        return ""
        
    async def evaluate(self, question: str, path: List[str], 
                      context: Optional[Dict[str, Any]] = None) -> Tuple[float, str]:
        """
        评估路径与问题的语义对齐度
        
        Args:
            question: 问题
            path: 关系路径
            context: 额外上下文
            
        Returns:
            (分数, 推理过程)
        """
        # 步骤1：提取语义约束
        constraints = await self._extract_constraints(question)
        
        # 步骤2：评估路径
        result = await self._evaluate_path(question, path, constraints, context)
        
        return result.score, result.reasoning
        
    async def _extract_constraints(self, question: str) -> Dict[str, Any]:
        """从问题中提取语义约束"""
        prompt = self.constraint_prompt.format(question=question)
        
        try:
            response = self.llm_model.generate_sentence(prompt)
            constraints = json.loads(response)
            
            # 验证是否为有效的JSON
            if not isinstance(constraints, dict):
                constraints = {}
                
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to extract constraints: {e}")
            constraints = {}
            
        return constraints
        
    async def _evaluate_path(self, question: str, path: List[str], 
                           constraints: Dict[str, Any],
                           context: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        """评估单条路径"""
        # 将路径转换为字符串表示
        path_str = self._format_path(path)
        constraints_json = json.dumps(constraints)
        
        # 构建评估提示词
        prompt = self.evaluator_prompt.format(
            question=question,
            constraints_json=constraints_json,
            path_str=path_str
        )
        
        # 如果有上下文，添加到提示词中
        if context:
            context_str = self._format_context(context)
            prompt = prompt.replace("**Task:**", f"**Additional Context:**\n{context_str}\n\n**Task:**")
            
        try:
            response = self.llm_model.generate_sentence(prompt)
            result = json.loads(response)
            
            score = float(result.get("score", 0.0))
            reasoning = result.get("reasoning", "No reasoning provided")
            
            # 确保分数在有效范围内
            score = max(0.0, min(1.0, score))
            
            # 应用评分调整：避免过于严格的评分
            adjusted_score = self._adjust_score(score, path, reasoning)
            
        except (json.JSONDecodeError, ValueError, Exception) as e:
            logger.warning(f"Failed to evaluate path: {e}")
            score = 0.0
            adjusted_score = 0.0
            reasoning = f"Evaluation failed: {str(e)}"
            
        return EvaluationResult(
            score=adjusted_score,
            reasoning=reasoning,
            constraints=constraints,
            confidence=self._compute_confidence(score, reasoning)
        )
        
    def _adjust_score(self, score: float, path: List[str], reasoning: str) -> float:
        """
        调整评分，避免过于严格的评分
        
        Args:
            score: 原始评分
            path: 路径
            reasoning: 评估理由
            
        Returns:
            调整后的评分
        """
        # 基本调整：提高低分但合理的路径的得分
        if 0.3 <= score <= 0.6:
            # 检查路径长度，短路径可能更可靠
            if len(path) <= 2:
                score += 0.1
                
            # 检查评估理由中的积极词汇
            positive_indicators = ["relevant", "related", "appropriate", "useful", "helpful", "correct", "valid"]
            if any(indicator in reasoning.lower() for indicator in positive_indicators):
                score += 0.1
                
        # 避免极端低分
        if score < 0.3 and len(path) > 0:
            score = max(0.3, score)  # 确保有内容的路径至少有0.3分
            
        # 确保分数在有效范围内
        return max(0.0, min(1.0, score))
        
    def _format_path(self, path: List[str]) -> str:
        """格式化路径为可读字符串"""
        if not path:
            return "[]"
        return " -> ".join(path)
        
    def _format_context(self, context: Dict[str, Any]) -> str:
        """格式化上下文信息"""
        parts = []
        
        if "neighbors" in context:
            parts.append(f"Neighboring paths: {context['neighbors'][:3]}")
            
        if "ancestors" in context:
            parts.append(f"Parent thoughts: {context['ancestors'][:2]}")
            
        if "metadata" in context and "iteration" in context["metadata"]:
            parts.append(f"Iteration: {context['metadata']['iteration']}")
            
        return "\n".join(parts)
        
    def _compute_confidence(self, score: float, reasoning: str) -> float:
        """计算评估的置信度"""
        # 基于推理长度和分数的置信度
        reasoning_length = len(reasoning.split())
        
        if reasoning_length < 10:
            confidence = 0.5
        elif reasoning_length < 30:
            confidence = 0.7
        else:
            confidence = 0.9
            
        # 极端分数降低置信度
        if score < 0.1 or score > 0.9:
            confidence *= 0.8
            
        return confidence
        
    async def batch_evaluate(self, question: str, paths: List[List[str]], 
                           batch_size: int = 5) -> List[Tuple[float, str]]:
        """批量评估多条路径"""
        results = []
        
        # 首先提取一次约束
        constraints = await self._extract_constraints(question)
        
        # 分批处理
        for i in range(0, len(paths), batch_size):
            batch = paths[i:i + batch_size]
            
            # 并行评估批次中的路径
            tasks = [
                self._evaluate_path(question, path, constraints)
                for path in batch
            ]
            
            batch_results = await asyncio.gather(*tasks)
            
            for result in batch_results:
                results.append((result.score, result.reasoning))
                
        return results
        
    async def compare_paths(self, question: str, path1: List[str], path2: List[str]) -> Dict[str, Any]:
        """比较两条路径的优劣"""
        # 分别评估
        score1, reasoning1 = await self.evaluate(question, path1)
        score2, reasoning2 = await self.evaluate(question, path2)
        
        # 生成比较分析
        comparison_prompt = f"""Compare these two reasoning paths for the question: "{question}"

Path 1: {self._format_path(path1)} (Score: {score1})
Reasoning: {reasoning1}

Path 2: {self._format_path(path2)} (Score: {score2})
Reasoning: {reasoning2}

Which path is better and why? Provide a brief analysis."""

        comparison = self.llm_model.generate_sentence(comparison_prompt)
        
        return {
            "path1": {"path": path1, "score": score1, "reasoning": reasoning1},
            "path2": {"path": path2, "score": score2, "reasoning": reasoning2},
            "comparison": comparison,
            "better_path": 1 if score1 > score2 else 2
        }