# src/got_reasoning/feedback_loop.py
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import json
import logging
from dataclasses import dataclass
from collections import defaultdict

from .thought_graph import ThoughtNode, ThoughtGraph
from utils import read_prompt

logger = logging.getLogger(__name__)


@dataclass
class FeedbackSignal:
    """反馈信号"""
    thought_id: str
    feedback_type: str  # improvement, refinement, correction, validation
    message: str
    suggestions: List[str]
    priority: float = 0.5
    
    
@dataclass
class ImprovementStrategy:
    """改进策略"""
    strategy_type: str  # extend, refine, decompose, combine
    description: str
    implementation: Dict[str, Any]
    expected_improvement: float
    

class FeedbackController:
    """反馈循环控制器：分析和改进思想"""
    
    def __init__(self, llm_model=None, window_size: int = 5):
        """
        Args:
            llm_model: LLM模型（可选）
            window_size: 历史窗口大小
        """
        self.llm_model = llm_model
        self.window_size = window_size
        
        # 历史记录
        self.history = {
            "scores": [],  # 每轮的最佳分数
            "improvements": [],  # 每轮的改进
            "failures": defaultdict(int)  # 失败模式计数
        }
        
        # 反馈策略
        self.strategies = {
            "low_score": self._handle_low_score,
            "stagnation": self._handle_stagnation,
            "divergence": self._handle_divergence,
            "incompleteness": self._handle_incompleteness,
            "contradiction": self._handle_contradiction
        }
        
    async def analyze(self, question: str, thoughts: List[ThoughtNode]) -> Dict[str, Any]:
        """
        分析当前思想集合并生成反馈
        
        Args:
            question: 原始问题
            thoughts: 当前思想列表
            
        Returns:
            反馈分析结果
        """
        analysis = {
            "status": "normal",
            "issues": [],
            "improvements": {},
            "global_feedback": ""
        }
        
        # 1. 检查整体性能
        if thoughts:
            scores = [t.score for t in thoughts]
            best_score = max(scores)
            avg_score = sum(scores) / len(scores)
            
            # 更新历史
            self.history["scores"].append(best_score)
            
            # 检查问题
            if best_score < 0.5:
                analysis["issues"].append("low_score")
                analysis["status"] = "problematic"
                
            if self._is_stagnating():
                analysis["issues"].append("stagnation")
                analysis["status"] = "stagnating"
                
            if self._is_diverging(scores):
                analysis["issues"].append("divergence")
                
        # 2. 分析个别思想
        for thought in thoughts:
            feedback = self._analyze_thought(thought, question)
            if feedback:
                analysis["improvements"][thought.id] = feedback
                
        # 3. 生成全局反馈
        if self.llm_model and analysis["issues"]:
            analysis["global_feedback"] = await self._generate_global_feedback(
                question, thoughts, analysis["issues"]
            )
            
        return analysis
        
    async def improve_thought(self, thought: ThoughtNode, 
                            feedback: FeedbackSignal) -> ThoughtNode:
        """
        根据反馈改进思想
        
        Args:
            thought: 原始思想
            feedback: 反馈信号
            
        Returns:
            改进后的思想
        """
        improvement_strategies = self._select_improvement_strategies(thought, feedback)
        
        # 应用改进策略
        improved_content = thought.content
        for strategy in improvement_strategies:
            improved_content = await self._apply_strategy(
                improved_content, 
                strategy, 
                thought.metadata
            )
            
        # 创建改进后的思想
        improved_thought = ThoughtNode(
            id=f"{thought.id}_improved",
            content=improved_content,
            node_type=thought.node_type,
            score=thought.score,  # 初始分数相同，后续会重新评估
            metadata={
                **thought.metadata,
                "improved_from": thought.id,
                "feedback_type": feedback.feedback_type,
                "improvements": [s.strategy_type for s in improvement_strategies]
            }
        )
        
        return improved_thought
        
    def _analyze_thought(self, thought: ThoughtNode, question: str) -> Optional[FeedbackSignal]:
        """分析单个思想"""
        issues = []
        suggestions = []
        
        # 检查思想质量
        if thought.score < 0.3:
            issues.append("very_low_score")
            suggestions.append("Consider alternative reasoning paths")
            
        # 检查思想完整性
        if isinstance(thought.content, list) and len(thought.content) < 2:
            issues.append("too_short")
            suggestions.append("Extend the reasoning chain")
            
        # 检查验证状态
        if thought.metadata.get("validated") == False:
            issues.append("validation_failed")
            suggestions.append("Ensure path exists in knowledge graph")
            
        if issues:
            return FeedbackSignal(
                thought_id=thought.id,
                feedback_type="improvement",
                message=f"Issues found: {', '.join(issues)}",
                suggestions=suggestions,
                priority=1.0 - thought.score
            )
            
        return None
        
    def _is_stagnating(self) -> bool:
        """检查是否停滞"""
        if len(self.history["scores"]) < self.window_size:
            return False
            
        recent_scores = self.history["scores"][-self.window_size:]
        
        # 检查分数是否几乎没有变化
        score_variance = np.var(recent_scores)
        return score_variance < 0.01
        
    def _is_diverging(self, scores: List[float]) -> bool:
        """检查是否发散"""
        if len(scores) < 2:
            return False
            
        # 检查分数分布是否过于分散
        score_std = np.std(scores)
        return score_std > 0.3
        
    def _handle_low_score(self, thoughts: List[ThoughtNode]) -> List[ImprovementStrategy]:
        """处理低分问题"""
        strategies = []
        
        # 策略1：扩展推理链
        strategies.append(ImprovementStrategy(
            strategy_type="extend",
            description="Add intermediate reasoning steps",
            implementation={"method": "add_bridge_relations"},
            expected_improvement=0.2
        ))
        
        # 策略2：分解问题
        strategies.append(ImprovementStrategy(
            strategy_type="decompose",
            description="Break down into simpler sub-problems",
            implementation={"method": "question_decomposition"},
            expected_improvement=0.15
        ))
        
        return strategies
        
    def _handle_stagnation(self, thoughts: List[ThoughtNode]) -> List[ImprovementStrategy]:
        """处理停滞问题"""
        strategies = []
        
        # 策略1：引入随机性
        strategies.append(ImprovementStrategy(
            strategy_type="diversify",
            description="Introduce randomness to explore new paths",
            implementation={"method": "random_perturbation", "strength": 0.3},
            expected_improvement=0.1
        ))
        
        # 策略2：重新开始
        strategies.append(ImprovementStrategy(
            strategy_type="restart",
            description="Start from different initial paths",
            implementation={"method": "alternative_initialization"},
            expected_improvement=0.25
        ))
        
        return strategies
        
    def _handle_divergence(self, thoughts: List[ThoughtNode]) -> List[ImprovementStrategy]:
        """处理发散问题"""
        strategies = []
        
        # 策略1：聚焦
        strategies.append(ImprovementStrategy(
            strategy_type="focus",
            description="Focus on most promising paths",
            implementation={"method": "top_k_selection", "k": 3},
            expected_improvement=0.15
        ))
        
        # 策略2：聚合相似思想
        strategies.append(ImprovementStrategy(
            strategy_type="merge_similar",
            description="Merge similar thoughts",
            implementation={"method": "similarity_clustering"},
            expected_improvement=0.2
        ))
        
        return strategies
        
    def _handle_incompleteness(self, thoughts: List[ThoughtNode]) -> List[ImprovementStrategy]:
        """处理不完整性问题"""
        return [
            ImprovementStrategy(
                strategy_type="complete",
                description="Fill in missing reasoning steps",
                implementation={"method": "gap_filling"},
                expected_improvement=0.3
            )
        ]
        
    def _handle_contradiction(self, thoughts: List[ThoughtNode]) -> List[ImprovementStrategy]:
        """处理矛盾问题"""
        return [
            ImprovementStrategy(
                strategy_type="resolve",
                description="Resolve contradictory paths",
                implementation={"method": "contradiction_resolution"},
                expected_improvement=0.25
            )
        ]
        
    def _select_improvement_strategies(self, thought: ThoughtNode, 
                                     feedback: FeedbackSignal) -> List[ImprovementStrategy]:
        """选择改进策略"""
        strategies = []
        
        # 根据反馈类型选择策略
        if "low_score" in feedback.message:
            strategies.extend(self._handle_low_score([thought]))
            
        if "too_short" in feedback.message:
            strategies.extend(self._handle_incompleteness([thought]))
            
        # 根据优先级排序
        strategies.sort(key=lambda s: s.expected_improvement, reverse=True)
        
        return strategies[:2]  # 最多应用2个策略
        
    async def _apply_strategy(self, content: Any, strategy: ImprovementStrategy, 
                            metadata: Dict[str, Any]) -> Any:
        """应用改进策略"""
        method = strategy.implementation["method"]
        
        if method == "add_bridge_relations":
            if isinstance(content, list):
                # 在路径中添加桥接关系
                # 这里需要更复杂的实现
                return content + ["bridge_relation"]
                
        elif method == "random_perturbation":
            if isinstance(content, list) and len(content) > 1:
                # 随机替换一个关系
                import random
                idx = random.randint(0, len(content) - 1)
                # 需要从可能的关系中选择
                return content
                
        # 其他策略的实现...
        
        return content
        
    async def _generate_global_feedback(self, question: str, thoughts: List[ThoughtNode], 
                                      issues: List[str]) -> str:
        """生成全局反馈"""
        if not self.llm_model:
            return "Multiple issues detected: " + ", ".join(issues)
            
        prompt = f"""Analyze the reasoning process for the question: "{question}"

Current issues: {', '.join(issues)}
Number of thoughts: {len(thoughts)}
Best score: {max(t.score for t in thoughts) if thoughts else 0}
Average score: {sum(t.score for t in thoughts) / len(thoughts) if thoughts else 0}

Provide constructive feedback on how to improve the reasoning process."""

        feedback = self.llm_model.generate_sentence(prompt)
        return feedback
        
    def reset_history(self):
        """重置历史记录"""
        self.history = {
            "scores": [],
            "improvements": [],
            "failures": defaultdict(int)
        }
        
    def get_statistics(self) -> Dict[str, Any]:
        """获取反馈统计信息"""
        stats = {
            "total_iterations": len(self.history["scores"]),
            "average_improvement": 0.0,
            "common_failures": [],
            "best_score": 0.0,
            "convergence_rate": 0.0
        }
        
        if self.history["scores"]:
            stats["best_score"] = max(self.history["scores"])
            
            # 计算平均改进
            if len(self.history["scores"]) > 1:
                improvements = [
                    self.history["scores"][i] - self.history["scores"][i-1]
                    for i in range(1, len(self.history["scores"]))
                ]
                stats["average_improvement"] = sum(improvements) / len(improvements)
                
            # 收敛率
            final_scores = self.history["scores"][-min(3, len(self.history["scores"])):]
            if len(final_scores) > 1:
                stats["convergence_rate"] = 1.0 - np.std(final_scores)
                
        # 常见失败模式
        stats["common_failures"] = sorted(
            self.history["failures"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return stats