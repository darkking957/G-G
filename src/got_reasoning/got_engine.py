# src/got_reasoning/got_engine.py
"""
GoT Engine - 完整版本，启用所有智能功能
"""

import asyncio
from typing import List, Dict, Any, Tuple, Optional, Set
import networkx as nx
from dataclasses import dataclass
import json
import torch
from tqdm import tqdm
import logging
import numpy as np
from collections import defaultdict

from .thought_graph import ThoughtGraph, ThoughtNode, ThoughtEdge
from .validate_plans import PlanValidator
from .evaluate_plans import SemanticEvaluator
from .aggregate_plans import ThoughtAggregator
from .graph_attention import GraphAttentionNetwork
from .feedback_loop import FeedbackController

logger = logging.getLogger(__name__)


@dataclass
class GoTConfig:
    """GoT引擎配置 - 默认启用完整功能"""
    max_iterations: int = 2  # 增加迭代次数
    beam_width: int = 10  # 增加beam宽度
    score_threshold: float = 0.5  # 降低阈值，允许更多探索
    enable_feedback: bool = True  # 默认启用反馈
    use_graph_attention: bool = True  # 默认启用GAT
    aggregation_strategy: str = "adaptive"  # 默认使用自适应聚合
    validation_mode: str = "relaxed"  # 使用宽松验证
    # 移除保守配置
    preserve_original: bool = False  # 不需要保留所有原始路径
    max_enhanced_paths: int = 20  # 允许更多增强路径
    enable_minimal_mode: bool = False  # 默认禁用最小化模式
    
    
class GoTEngine:
    """Graph of Thoughts推理引擎 - 完整实现"""
    
    def __init__(self, config: GoTConfig, llm_model, kg_graph):
        self.config = config
        self.llm_model = llm_model
        self.kg_graph = kg_graph
        
        # 初始化所有组件
        self._init_components()
            
    def _init_components(self):
        """初始化所有组件"""
        self.thought_graph = ThoughtGraph()
        self.validator = PlanValidator(self.kg_graph, validation_mode=self.config.validation_mode)
        self.evaluator = SemanticEvaluator(self.llm_model)
        self.aggregator = ThoughtAggregator(self.llm_model)
        self.feedback_controller = FeedbackController(self.llm_model)
        
        if self.config.use_graph_attention:
            self.gat = GraphAttentionNetwork(
                input_dim=768,
                hidden_dim=256,
                output_dim=128,
                num_heads=8
            )
    
    async def reason(self, question: str, initial_paths: List[List[str]]) -> Dict[str, Any]:
        """
        执行完整的GoT推理
        """
        logger.info(f"Starting full GoT reasoning for: {question}")
        
        # 阶段1: 初始化思想图
        await self._initialize_thought_graph(question, initial_paths)
        
        # 阶段2: 结构验证与修复
        valid_thoughts = await self._validate_and_repair_thoughts()
        
        # 阶段3: 迭代推理（GoT核心）
        best_thoughts = await self._iterative_reasoning(question, valid_thoughts)
        
        # 阶段4: 生成最终答案
        result = await self._generate_final_answer(question, best_thoughts)
        
        return result
    
    async def _initialize_thought_graph(self, question: str, initial_paths: List[List[str]]):
        """初始化思想图"""
        # 创建根节点
        root = ThoughtNode(
            id="root",
            content=question,
            node_type="question",
            score=1.0
        )
        self.thought_graph.add_node(root)
        
        # 为每个初始路径创建思想节点
        for i, path in enumerate(initial_paths):
            thought = ThoughtNode(
                id=f"initial_{i}",
                content=path,
                node_type="path",
                score=0.0,
                metadata={"depth": 1, "source": "initial", "validated": False}
            )
            self.thought_graph.add_node(thought)
            self.thought_graph.add_edge(ThoughtEdge(
                source=root.id,
                target=thought.id,
                edge_type="generates"
            ))
    
    async def _validate_and_repair_thoughts(self) -> List[ThoughtNode]:
        """验证并修复思想节点"""
        logger.info("Phase 2: Validating and repairing paths")
        
        path_thoughts = self.thought_graph.get_nodes_by_type("path")
        valid_thoughts = []
        
        # 并行验证
        validation_tasks = []
        for thought in path_thoughts:
            task = self.validator.validate_path(thought.content)
            validation_tasks.append((thought, task))
        
        # 收集结果
        for thought, task in validation_tasks:
            is_valid = await task
            thought.metadata["validated"] = is_valid
            
            if is_valid:
                valid_thoughts.append(thought)
            else:
                # 尝试修复无效路径
                repaired_path, confidence = self.validator.repair_path(thought.content)
                if confidence > 0.5:
                    repaired_thought = ThoughtNode(
                        id=f"repaired_{thought.id}",
                        content=repaired_path,
                        node_type="path",
                        score=confidence * 0.7,
                        metadata={"validated": True, "repaired": True, "parent": thought.id}
                    )
                    self.thought_graph.add_node(repaired_thought)
                    valid_thoughts.append(repaired_thought)
                    
        logger.info(f"Validated {len(valid_thoughts)} out of {len(path_thoughts)} paths")
        return valid_thoughts
    
    async def _iterative_reasoning(self, question: str, valid_thoughts: List[ThoughtNode]) -> List[ThoughtNode]:
        """迭代推理过程 - GoT核心"""
        logger.info("Phase 3: Iterative reasoning")
        
        current_thoughts = valid_thoughts
        
        for iteration in range(self.config.max_iterations):
            logger.info(f"Iteration {iteration + 1}/{self.config.max_iterations}")
            
            # 步骤1: 语义评估
            await self._evaluate_thoughts(question, current_thoughts)
            
            # 步骤2: 思想聚合
            if self.config.aggregation_strategy != "none" and iteration < self.config.max_iterations - 1:
                new_thoughts = await self._aggregate_thoughts(question, current_thoughts)
                current_thoughts.extend(new_thoughts)
            
            # 步骤3: 图注意力增强（如果启用）
            if self.config.use_graph_attention and len(current_thoughts) > 5:
                await self._apply_graph_attention(current_thoughts)
            
            # 步骤4: 反馈循环（如果启用）
            if self.config.enable_feedback:
                improved_thoughts = await self._apply_feedback(question, current_thoughts)
                current_thoughts.extend(improved_thoughts)
            
            # 步骤5: 筛选和剪枝
            current_thoughts = self._select_best_thoughts(current_thoughts)
            
            # 检查是否收敛
            if self._check_convergence(current_thoughts, iteration):
                logger.info(f"Converged at iteration {iteration + 1}")
                break
        
        return current_thoughts
    
    async def _evaluate_thoughts(self, question: str, thoughts: List[ThoughtNode]):
            """评估思想的语义相关性"""
            logger.info(f"Evaluating {len(thoughts)} thoughts")
            
            # 步骤1: 收集所有需要评估的思想，并为它们创建协程任务
            thoughts_to_evaluate = []
            evaluation_coroutines = []
            for thought in thoughts:
                if thought.node_type == "path" and thought.score == 0:  # 只评估未评分的
                    thoughts_to_evaluate.append(thought)
                    # 创建协程，但不要在这里 await
                    coroutine = self.evaluator.evaluate(question, thought.content)
                    evaluation_coroutines.append(coroutine)

            # 如果没有需要评估的任务，直接返回
            if not evaluation_coroutines:
                return

            # 步骤2: 使用 asyncio.gather 并发执行所有评估任务
            # gather 会返回一个包含所有任务结果的列表，顺序与输入协程的顺序一致
            logger.info(f"Running {len(evaluation_coroutines)} evaluation tasks in parallel...")
            results = await asyncio.gather(*evaluation_coroutines)

            # 步骤3: 将返回的结果（(score, reasoning)元组）赋值给对应的思想节点
            for i, thought in enumerate(thoughts_to_evaluate):
                score, reasoning = results[i]
                thought.score = score
                thought.metadata["eval_reasoning"] = reasoning
    
    async def _aggregate_thoughts(self, question: str, thoughts: List[ThoughtNode]) -> List[ThoughtNode]:
        """聚合思想生成新的复合思想"""
        logger.info("Aggregating thoughts")
        
        # 选择高分思想进行聚合
        high_score_thoughts = [t for t in thoughts if t.score > 0.6]
        
        if len(high_score_thoughts) < 2:
            return []
        
        # 根据策略选择聚合方法
        if self.config.aggregation_strategy == "adaptive":
            # 分析聚合潜力
            potential = self.aggregator.analyze_aggregation_potential(high_score_thoughts)
            if potential["potential"] == "low":
                return []
            strategy = "greedy" if potential["potential"] == "medium" else "exhaustive"
        else:
            strategy = self.config.aggregation_strategy
        
        # 执行聚合
        new_thoughts = await self.aggregator.multi_aggregate(
            question, 
            high_score_thoughts[:10],  # 限制数量
            strategy=strategy
        )
        
        # 将新思想添加到图中
        for thought in new_thoughts:
            self.thought_graph.add_node(thought)
            
        logger.info(f"Generated {len(new_thoughts)} new aggregated thoughts")
        return new_thoughts
    
    async def _apply_graph_attention(self, thoughts: List[ThoughtNode]):
        """应用图注意力机制增强思想表示"""
        if len(thoughts) < 5:  # 思想太少，不值得使用GAT
            return
            
        logger.info("Applying graph attention")
        
        # 获取思想嵌入
        embeddings = {}
        for thought in thoughts:
            # 这里简化处理，实际应该使用LLM获取嵌入
            embeddings[thought.id] = await self._get_thought_embedding(thought)
        
        # 创建邻接矩阵
        adj_matrix = self.thought_graph.get_adjacency_matrix()
        
        # 应用GAT（这里简化处理）
        # 实际实现需要更复杂的处理
        for thought in thoughts:
            if thought.id in embeddings:
                # 模拟GAT效果：基于邻居调整分数
                neighbors = self.thought_graph.get_neighbors(thought.id)
                if neighbors:
                    neighbor_scores = [n.score for n in neighbors]
                    # 使用注意力机制的简化版本
                    attention_boost = np.mean(neighbor_scores) * 0.1
                    thought.score = min(1.0, thought.score + attention_boost)
    
    async def _apply_feedback(self, question: str, thoughts: List[ThoughtNode]) -> List[ThoughtNode]:
        """应用反馈循环改进思想"""
        logger.info("Applying feedback loop")
        
        # 分析当前状态
        analysis = await self.feedback_controller.analyze(question, thoughts)
        
        if analysis["status"] == "normal":
            return []
        
        # 根据分析结果改进思想
        improved_thoughts = []
        
        for thought_id, feedback in analysis["improvements"].items():
            thought = self.thought_graph.get_node(thought_id)
            if thought:
                improved = await self.feedback_controller.improve_thought(thought, feedback)
                # 重新评估改进后的思想
                score, reasoning = await self.evaluator.evaluate(question, improved.content)
                improved.score = score
                improved.metadata["improved"] = True
                improved_thoughts.append(improved)
                self.thought_graph.add_node(improved)
        
        logger.info(f"Generated {len(improved_thoughts)} improved thoughts")
        return improved_thoughts
    
    def _select_best_thoughts(self, thoughts: List[ThoughtNode]) -> List[ThoughtNode]:
        """选择最佳思想"""
        # 按分数排序
        sorted_thoughts = sorted(thoughts, key=lambda t: t.score, reverse=True)
        
        # 去重：如果内容相同，保留分数最高的
        unique_thoughts = {}
        for thought in sorted_thoughts:
            content_key = tuple(thought.content) if isinstance(thought.content, list) else thought.content
            if content_key not in unique_thoughts or thought.score > unique_thoughts[content_key].score:
                unique_thoughts[content_key] = thought
        
        # 返回前N个
        return list(unique_thoughts.values())[:self.config.beam_width]
    
    def _check_convergence(self, thoughts: List[ThoughtNode], iteration: int) -> bool:
        """检查是否收敛"""
        if not thoughts:
            return True
            
        # 如果最高分已经很高
        max_score = max(t.score for t in thoughts)
        if max_score > 0.95:
            return True
            
        # 如果分数不再提升
        if hasattr(self, '_prev_max_score'):
            if abs(max_score - self._prev_max_score) < 0.01:
                return True
                
        self._prev_max_score = max_score
        return False
    
    async def _generate_final_answer(self, question: str, best_thoughts: List[ThoughtNode]) -> Dict[str, Any]:
        """生成最终答案"""
        logger.info("Generating final answer")
        
        # 选择最佳思想路径
        if best_thoughts:
            best_thought = best_thoughts[0]
            best_paths = [t.content for t in best_thoughts[:5]]  # 返回前5个
        else:
            best_thought = None
            best_paths = []
        
        # 构建详细结果
        result = {
            "question": question,
            "best_path": best_thought.content if best_thought else [],
            "score": best_thought.score if best_thought else 0.0,
            "reasoning_paths": best_paths,
            "thought_graph": self.thought_graph if not self.config.enable_minimal_mode else None,
            "iterations": self.config.max_iterations,
            "total_thoughts": len(self.thought_graph.nodes),
            "statistics": {
                "validated_thoughts": len([n for n in self.thought_graph.nodes.values() 
                                         if n.metadata.get("validated", False)]),
                "aggregated_thoughts": len([n for n in self.thought_graph.nodes.values() 
                                          if n.node_type == "aggregated"]),
                "improved_thoughts": len([n for n in self.thought_graph.nodes.values() 
                                        if n.metadata.get("improved", False)]),
            }
        }
        
        # 添加推理解释
        if best_thought and "eval_reasoning" in best_thought.metadata:
            result["explanation"] = best_thought.metadata["eval_reasoning"]
        
        return result
    
    async def _get_thought_embedding(self, thought: ThoughtNode) -> torch.Tensor:
        """获取思想的嵌入表示"""
        # 这里应该使用LLM获取真实的嵌入
        # 简化实现：基于内容生成伪嵌入
        if isinstance(thought.content, list):
            # 基于路径长度和内容生成特征
            features = torch.zeros(768)
            features[0] = len(thought.content)  # 路径长度
            features[1] = thought.score  # 当前分数
            # 添加一些随机性
            features[2:] = torch.randn(766) * 0.1
            return features
        else:
            # 对于其他类型的内容
            return torch.randn(768)
    
    async def explain_reasoning(self, result: Dict[str, Any]) -> str:
        """生成推理过程的解释"""
        explanation = f"For the question '{result['question']}':\n\n"
        
        if result["best_path"]:
            explanation += f"The best reasoning path found is: {' -> '.join(result['best_path'])}\n"
            explanation += f"With confidence score: {result['score']:.3f}\n\n"
            
            # 解释推理过程
            stats = result.get("statistics", {})
            explanation += "Reasoning process:\n"
            explanation += f"- Started with {result['total_thoughts']} initial paths\n"
            explanation += f"- Validated {stats.get('validated_thoughts', 0)} paths\n"
            
            if stats.get("aggregated_thoughts", 0) > 0:
                explanation += f"- Created {stats['aggregated_thoughts']} combined paths\n"
                
            if stats.get("improved_thoughts", 0) > 0:
                explanation += f"- Improved {stats['improved_thoughts']} paths through feedback\n"
                
            explanation += f"- Selected top {len(result['reasoning_paths'])} paths after {result['iterations']} iterations\n"
        else:
            explanation += "No valid reasoning path found.\n"
            
        return explanation