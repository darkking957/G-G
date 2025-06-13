# src/got_reasoning/aggregate_plans.py
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

from typing import List, Dict, Any, Tuple, Optional, Set
import json
import logging
from dataclasses import dataclass
import itertools

from utils import read_prompt, build_graph, bfs_with_rule
from .thought_graph import ThoughtNode

logger = logging.getLogger(__name__)


@dataclass
class AggregationResult:
    """聚合结果"""
    content: Any  # 聚合后的内容（路径或其他）
    type: str  # 聚合类型：sequential, parallel, hierarchical
    confidence: float
    explanation: str
    

class ThoughtAggregator:
    """思想聚合器：将多个思想组合成更强大的复合思想"""
    
    def __init__(self, llm_model, prompt_dir: str = "prompts"):
        """
        Args:
            llm_model: LLM模型实例
            prompt_dir: 提示词模板目录
        """
        self.llm_model = llm_model
        self.prompt_dir = prompt_dir
        
        # 加载提示词模板
        self.aggregator_prompt = self._load_prompt("aggregator_prompt.txt")
        
    def _load_prompt(self, filename: str) -> str:
        """加载提示词模板"""
        try:
            return read_prompt(os.path.join(self.prompt_dir, filename))
        except FileNotFoundError:
            return self._get_default_prompt(filename)
            
    def _get_default_prompt(self, filename: str) -> str:
        """获取默认提示词模板"""
        if filename == "aggregator_prompt.txt":
            return """**Context:**
- User Question: "{question}"
- Thought 1: {thought1_str}
- Thought 2: {thought2_str}

**Task:**
Analyze how these two thoughts can be combined to better answer the question.
Consider different aggregation strategies:

1. **Sequential**: Connect thoughts in sequence (A leads to B)
2. **Parallel**: Use both thoughts independently to get different aspects
3. **Hierarchical**: One thought refines or specializes the other
4. **Union**: Merge overlapping information
5. **Intersection**: Focus on common elements

Synthesize these thoughts into a single, coherent reasoning chain.
Output in JSON format:
{{
    "aggregated_path": <list or string>,
    "aggregation_type": "<type>",
    "explanation": "<why this aggregation helps>",
    "confidence": <0.0-1.0>
}}"""
        
        return ""
        
    async def aggregate(self, question: str, thought1: ThoughtNode, 
                       thought2: ThoughtNode) -> Optional[Dict[str, Any]]:
        """
        聚合两个思想节点
        
        Args:
            question: 原始问题
            thought1: 第一个思想节点
            thought2: 第二个思想节点
            
        Returns:
            聚合结果或None（如果无法聚合）
        """
        # 检查是否可以聚合
        if not self._can_aggregate(thought1, thought2):
            return None
            
        # 根据思想类型选择聚合策略
        if thought1.node_type == "path" and thought2.node_type == "path":
            return await self._aggregate_paths(question, thought1, thought2)
        else:
            return await self._aggregate_general(question, thought1, thought2)
            
    def _can_aggregate(self, thought1: ThoughtNode, thought2: ThoughtNode) -> bool:
        """判断两个思想是否可以聚合"""
        # 避免聚合相同的思想
        if thought1.id == thought2.id:
            return False
            
        # 避免聚合分数都很低的思想
        if thought1.score < 0.3 and thought2.score < 0.3:
            return False
            
        # 其他可聚合性检查...
        return True
        
    async def _aggregate_paths(self, question: str, thought1: ThoughtNode, 
                             thought2: ThoughtNode) -> Optional[Dict[str, Any]]:
        """聚合两条路径"""
        path1 = thought1.content
        path2 = thought2.content
        
        # 检查路径的可连接性
        connection_type = self._check_path_connection(path1, path2)
        
        if connection_type == "sequential":
            # 顺序连接
            aggregated_path = path1 + path2
            aggregation_type = "sequential"
            
        elif connection_type == "overlapping":
            # 有重叠，需要合并
            aggregated_path = self._merge_overlapping_paths(path1, path2)
            aggregation_type = "union"
            
        else:
            # 使用LLM决定聚合策略
            result = await self._llm_aggregate(question, thought1, thought2)
            if not result:
                return None
                
            aggregated_path = result.content
            aggregation_type = result.type
            
        return {
            "content": aggregated_path,
            "type": aggregation_type,
            "confidence": (thought1.score + thought2.score) / 2
        }
        
    async def _aggregate_general(self, question: str, thought1: ThoughtNode, 
                               thought2: ThoughtNode) -> Optional[Dict[str, Any]]:
        """聚合一般类型的思想"""
        result = await self._llm_aggregate(question, thought1, thought2)
        
        if result:
            return {
                "content": result.content,
                "type": result.type,
                "confidence": result.confidence
            }
            
        return None
        
    async def _llm_aggregate(self, question: str, thought1: ThoughtNode, 
                           thought2: ThoughtNode) -> Optional[AggregationResult]:
        """使用LLM进行聚合"""
        # 格式化思想内容
        thought1_str = self._format_thought(thought1)
        thought2_str = self._format_thought(thought2)
        
        # 构建提示词
        prompt = self.aggregator_prompt.format(
            question=question,
            thought1_str=thought1_str,
            thought2_str=thought2_str
        )
        
        try:
            response = self.llm_model.generate_sentence(prompt)
            result = json.loads(response)
            
            return AggregationResult(
                content=result.get("aggregated_path", []),
                type=result.get("aggregation_type", "unknown"),
                confidence=float(result.get("confidence", 0.5)),
                explanation=result.get("explanation", "")
            )
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to aggregate thoughts: {e}")
            return None
            
    def _check_path_connection(self, path1: List[str], path2: List[str]) -> str:
        """检查两条路径的连接关系"""
        if not path1 or not path2:
            return "none"
            
        # 检查是否可以顺序连接
        # 这需要更复杂的逻辑，比如检查path1的终点是否可以作为path2的起点
        # 简化实现
        
        # 检查是否有重叠
        set1 = set(path1)
        set2 = set(path2)
        if set1 & set2:
            return "overlapping"
            
        return "independent"
        
    def _merge_overlapping_paths(self, path1: List[str], path2: List[str]) -> List[str]:
        """合并有重叠的路径"""
        # 简单实现：去重合并
        seen = set()
        merged = []
        
        for rel in path1 + path2:
            if rel not in seen:
                seen.add(rel)
                merged.append(rel)
                
        return merged
        
    def _format_thought(self, thought: ThoughtNode) -> str:
        """格式化思想内容为字符串"""
        if isinstance(thought.content, list):
            return f"Path: {' -> '.join(thought.content)}"
        else:
            return str(thought.content)
            
    async def multi_aggregate(self, question: str, thoughts: List[ThoughtNode], 
                            strategy: str = "greedy") -> List[ThoughtNode]:
        """
        聚合多个思想
        
        Args:
            question: 问题
            thoughts: 思想列表
            strategy: 聚合策略 - greedy, exhaustive, hierarchical
            
        Returns:
            聚合后的新思想列表
        """
        new_thoughts = []
        
        if strategy == "greedy":
            # 贪心策略：按分数排序，优先聚合高分思想
            sorted_thoughts = sorted(thoughts, key=lambda t: t.score, reverse=True)
            
            used = set()
            for i, t1 in enumerate(sorted_thoughts):
                if t1.id in used:
                    continue
                    
                for j, t2 in enumerate(sorted_thoughts[i+1:], i+1):
                    if t2.id in used:
                        continue
                        
                    result = await self.aggregate(question, t1, t2)
                    if result:
                        new_thought = ThoughtNode(
                            id=f"multi_agg_{len(new_thoughts)}",
                            content=result["content"],
                            node_type="aggregated",
                            score=result["confidence"],
                            metadata={
                                "source_ids": [t1.id, t2.id],
                                "aggregation_type": result["type"]
                            }
                        )
                        new_thoughts.append(new_thought)
                        used.add(t1.id)
                        used.add(t2.id)
                        break
                        
        elif strategy == "exhaustive":
            # 穷举策略：尝试所有可能的组合
            pairs = list(itertools.combinations(thoughts, 2))
            for t1, t2 in pairs[:10]:  # 限制数量
                result = await self.aggregate(question, t1, t2)
                if result:
                    new_thought = ThoughtNode(
                        id=f"exhaustive_agg_{len(new_thoughts)}",
                        content=result["content"],
                        node_type="aggregated",
                        score=result["confidence"],
                        metadata={
                            "source_ids": [t1.id, t2.id],
                            "aggregation_type": result["type"]
                        }
                    )
                    new_thoughts.append(new_thought)
                    
        elif strategy == "hierarchical":
            # 层次策略：基于思想之间的关系进行聚合
            # 需要分析思想图的结构
            pass
            
        return new_thoughts
        
    def analyze_aggregation_potential(self, thoughts: List[ThoughtNode]) -> Dict[str, Any]:
        """分析思想集合的聚合潜力"""
        n = len(thoughts)
        if n < 2:
            return {"potential": "low", "reason": "Too few thoughts"}
            
        # 分析思想的多样性
        types = set(t.node_type for t in thoughts)
        scores = [t.score for t in thoughts]
        
        # 计算聚合潜力指标
        diversity = len(types) / n
        score_variance = sum((s - sum(scores)/n)**2 for s in scores) / n
        
        if diversity > 0.5 and score_variance > 0.1:
            potential = "high"
            reason = "High diversity and score variance"
        elif diversity > 0.3 or score_variance > 0.05:
            potential = "medium"
            reason = "Moderate diversity or score variance"
        else:
            potential = "low"
            reason = "Low diversity and similar scores"
            
        return {
            "potential": potential,
            "reason": reason,
            "metrics": {
                "diversity": diversity,
                "score_variance": score_variance,
                "num_thoughts": n,
                "thought_types": list(types)
            }
        }