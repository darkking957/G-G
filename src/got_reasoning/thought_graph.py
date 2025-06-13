# src/got_reasoning/thought_graph.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
import networkx as nx
import json
from datetime import datetime
import uuid
import torch

@dataclass
class ThoughtNode:
    """思想节点"""
    id: str
    content: Any  # 可以是路径列表、文本或其他
    node_type: str  # question, path, aggregated, refined
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "node_type": self.node_type,
            "score": self.score,
            "metadata": self.metadata,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThoughtNode':
        return cls(**data)


@dataclass
class ThoughtEdge:
    """思想边"""
    source: str
    target: str
    edge_type: str  # generates, aggregated_to, refined_from, validates
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type,
            "weight": self.weight,
            "metadata": self.metadata
        }


class ThoughtGraph:
    """思想图数据结构"""
    
    def __init__(self):
        self.nodes: Dict[str, ThoughtNode] = {}
        self.edges: List[ThoughtEdge] = []
        self.graph = nx.DiGraph()
        
    def add_node(self, node: ThoughtNode):
        """添加节点"""
        self.nodes[node.id] = node
        self.graph.add_node(node.id, data=node)
        
    def add_edge(self, edge: ThoughtEdge):
        """添加边"""
        self.edges.append(edge)
        self.graph.add_edge(
            edge.source, 
            edge.target, 
            edge_type=edge.edge_type,
            weight=edge.weight,
            data=edge
        )
        
    def get_node(self, node_id: str) -> Optional[ThoughtNode]:
        """获取节点"""
        return self.nodes.get(node_id)
        
    def get_nodes_by_type(self, node_type: str) -> List[ThoughtNode]:
        """根据类型获取节点"""
        return [node for node in self.nodes.values() if node.node_type == node_type]
        
    def get_neighbors(self, node_id: str) -> List[ThoughtNode]:
        """获取邻居节点"""
        neighbor_ids = list(self.graph.neighbors(node_id))
        return [self.nodes[nid] for nid in neighbor_ids if nid in self.nodes]
        
    def get_ancestors(self, node_id: str) -> List[ThoughtNode]:
        """获取祖先节点"""
        ancestors = []
        for ancestor_id in nx.ancestors(self.graph, node_id):
            if ancestor_id in self.nodes:
                ancestors.append(self.nodes[ancestor_id])
        return ancestors
        
    def get_descendants(self, node_id: str) -> List[ThoughtNode]:
        """获取后代节点"""
        descendants = []
        for desc_id in nx.descendants(self.graph, node_id):
            if desc_id in self.nodes:
                descendants.append(self.nodes[desc_id])
        return descendants
        
    def get_path(self, source_id: str, target_id: str) -> Optional[List[ThoughtNode]]:
        """获取两个节点之间的路径"""
        try:
            path_ids = nx.shortest_path(self.graph, source_id, target_id)
            return [self.nodes[nid] for nid in path_ids]
        except (nx.NetworkXNoPath, KeyError):
            return None
            
    def get_subgraph(self, node_ids: List[str]) -> 'ThoughtGraph':
        """获取子图"""
        subgraph = ThoughtGraph()
        
        # 添加节点
        for node_id in node_ids:
            if node_id in self.nodes:
                subgraph.add_node(self.nodes[node_id])
        
        # 添加边
        for edge in self.edges:
            if edge.source in node_ids and edge.target in node_ids:
                subgraph.add_edge(edge)
                
        return subgraph
        
    def merge_thoughts(self, thought_ids: List[str], merge_strategy: str = "union") -> ThoughtNode:
        """合并多个思想节点"""
        thoughts = [self.nodes[tid] for tid in thought_ids if tid in self.nodes]
        
        if not thoughts:
            raise ValueError("No valid thoughts to merge")
            
        # 创建合并后的节点
        merged_id = f"merged_{uuid.uuid4().hex[:8]}"
        
        if merge_strategy == "union":
            # 合并内容（假设是路径列表）
            merged_content = []
            for thought in thoughts:
                if isinstance(thought.content, list):
                    merged_content.extend(thought.content)
                else:
                    merged_content.append(thought.content)
            
            # 去重
            if all(isinstance(item, list) for item in merged_content):
                # 路径去重
                unique_content = []
                seen = set()
                for path in merged_content:
                    path_tuple = tuple(path)
                    if path_tuple not in seen:
                        seen.add(path_tuple)
                        unique_content.append(path)
                merged_content = unique_content
        else:
            # 其他合并策略
            merged_content = [t.content for t in thoughts]
            
        merged_node = ThoughtNode(
            id=merged_id,
            content=merged_content,
            node_type="merged",
            score=max(t.score for t in thoughts),  # 使用最高分
            metadata={
                "source_ids": thought_ids,
                "merge_strategy": merge_strategy,
                "source_scores": [t.score for t in thoughts]
            }
        )
        
        # 添加到图中
        self.add_node(merged_node)
        
        # 添加合并边
        for thought_id in thought_ids:
            self.add_edge(ThoughtEdge(
                source=thought_id,
                target=merged_id,
                edge_type="merged_to"
            ))
            
        return merged_node
        
    def prune_low_score_branches(self, threshold: float = 0.3):
        """剪枝低分支路"""
        nodes_to_remove = []
        
        for node_id, node in self.nodes.items():
            # 保留根节点和高分节点
            if node.node_type == "question" or node.score >= threshold:
                continue
                
            # 检查是否有高分后代
            descendants = self.get_descendants(node_id)
            if not any(d.score >= threshold for d in descendants):
                nodes_to_remove.append(node_id)
                
        # 移除节点
        for node_id in nodes_to_remove:
            self.remove_node(node_id)
            
    def remove_node(self, node_id: str):
        """移除节点及相关边"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            self.graph.remove_node(node_id)
            
            # 移除相关边
            self.edges = [e for e in self.edges 
                         if e.source != node_id and e.target != node_id]
            
    def get_adjacency_matrix(self, node_ids: Optional[List[str]] = None) -> torch.Tensor:
        """获取邻接矩阵"""
        import torch
        
        if node_ids is None:
            node_ids = list(self.nodes.keys())
            
        n = len(node_ids)
        node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        
        adj_matrix = torch.zeros((n, n))
        
        for edge in self.edges:
            if edge.source in node_id_to_idx and edge.target in node_id_to_idx:
                i = node_id_to_idx[edge.source]
                j = node_id_to_idx[edge.target]
                adj_matrix[i, j] = edge.weight
                
        return adj_matrix
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges]
        }
        
    def to_json(self) -> str:
        """转换为JSON"""
        return json.dumps(self.to_dict(), indent=2)
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThoughtGraph':
        """从字典创建"""
        graph = cls()
        
        # 添加节点
        for node_data in data.get("nodes", []):
            node = ThoughtNode.from_dict(node_data)
            graph.add_node(node)
            
        # 添加边
        for edge_data in data.get("edges", []):
            edge = ThoughtEdge(**edge_data)
            graph.add_edge(edge)
            
        return graph
        
    def visualize(self, output_path: str = "thought_graph.png"):
        """可视化思想图"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        # 设置节点颜色
        node_colors = []
        for node_id in self.graph.nodes():
            node = self.nodes.get(node_id)
            if node:
                if node.node_type == "question":
                    node_colors.append("lightblue")
                elif node.node_type == "path":
                    node_colors.append("lightgreen")
                elif node.node_type == "aggregated":
                    node_colors.append("orange")
                else:
                    node_colors.append("lightgray")
            else:
                node_colors.append("lightgray")
                
        # 设置节点大小（基于分数）
        node_sizes = []
        for node_id in self.graph.nodes():
            node = self.nodes.get(node_id)
            if node:
                size = 300 + node.score * 700  # 300-1000
                node_sizes.append(size)
            else:
                node_sizes.append(300)
                
        # 布局
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        
        # 绘制
        nx.draw(
            self.graph,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            with_labels=True,
            font_size=8,
            font_weight='bold',
            arrows=True,
            edge_color='gray',
            alpha=0.7
        )
        
        # 添加边标签
        edge_labels = {}
        for edge in self.edges:
            edge_labels[(edge.source, edge.target)] = edge.edge_type
            
        nx.draw_networkx_edge_labels(
            self.graph,
            pos,
            edge_labels,
            font_size=6
        )
        
        plt.title("Thought Graph Visualization")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def get_statistics(self) -> Dict[str, Any]:
        """获取图的统计信息"""
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "node_types": {
                node_type: len(self.get_nodes_by_type(node_type))
                for node_type in set(n.node_type for n in self.nodes.values())
            },
            "avg_score": sum(n.score for n in self.nodes.values()) / len(self.nodes) if self.nodes else 0,
            "max_score": max((n.score for n in self.nodes.values()), default=0),
            "graph_density": nx.density(self.graph),
            "is_dag": nx.is_directed_acyclic_graph(self.graph)
        }

    def get_best_path_nodes(self, min_score: float = 0.0) -> List[ThoughtNode]:
        """
        获取最佳路径节点，即使它们未通过验证
        
        Args:
            min_score: 最小分数阈值
            
        Returns:
            按分数排序的路径节点列表
        """
        # 获取所有路径类型的节点
        path_nodes = self.get_nodes_by_type("path")
        
        # 过滤掉低于分数阈值的节点
        filtered_nodes = [node for node in path_nodes if node.score >= min_score]
        
        # 按分数排序
        sorted_nodes = sorted(filtered_nodes, key=lambda n: n.score, reverse=True)
        
        return sorted_nodes

    def get_partial_valid_paths(self) -> List[ThoughtNode]:
        """
        获取部分有效的路径节点
        
        Returns:
            部分有效的路径节点列表
        """
        # 获取所有路径类型的节点
        path_nodes = self.get_nodes_by_type("path")
        
        # 过滤出部分有效的节点
        partial_valid_nodes = []
        
        for node in path_nodes:
            # 检查节点元数据中的验证信息
            if node.metadata.get("partially_valid", False):
                partial_valid_nodes.append(node)
            
        # 按分数排序
        sorted_nodes = sorted(partial_valid_nodes, key=lambda n: n.score, reverse=True)
        
        return sorted_nodes

    def get_most_promising_nodes(self, top_k: int = 3) -> List[ThoughtNode]:
        """
        获取最有希望的节点，综合考虑分数、验证状态和节点类型
        
        Args:
            top_k: 返回的节点数量
            
        Returns:
            最有希望的节点列表
        """
        # 获取所有非问题类型的节点
        nodes = [node for node in self.nodes.values() if node.node_type != "question"]
        
        if not nodes:
            return []
        
        # 定义节点排序函数
        def node_rank(node):
            # 基础分数
            base_score = node.score
            
            # 验证状态加分
            validation_bonus = 0.3 if node.metadata.get("validated", False) else 0
            
            # 部分有效加分
            partial_bonus = 0.1 if node.metadata.get("partially_valid", False) else 0
            
            # 聚合节点加分
            aggregation_bonus = 0.2 if node.node_type == "aggregated" else 0
            
            # 迭代轮次加分（后期迭代的节点可能更好）
            iteration = node.metadata.get("iteration", 0)
            iteration_bonus = min(0.1, iteration * 0.02)
            
            return base_score + validation_bonus + partial_bonus + aggregation_bonus + iteration_bonus
        
        # 按综合排序函数排序
        sorted_nodes = sorted(nodes, key=node_rank, reverse=True)
        
        return sorted_nodes[:top_k]