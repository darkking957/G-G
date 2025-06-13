# src/got_reasoning/validate_plans.py
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

from typing import List, Dict, Any, Tuple, Optional, Set
import networkx as nx
from collections import deque
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

from utils.graph_utils import build_graph, bfs_with_rule

logger = logging.getLogger(__name__)


class PlanValidator:
    """路径验证器：验证路径在知识图谱中的存在性"""
    
    def __init__(self, kg_graph: nx.Graph, max_workers: int = 4, validation_mode: str = "relaxed"):
        """
        Args:
            kg_graph: 知识图谱
            max_workers: 并行验证的最大工作线程数
            validation_mode: 验证模式，"strict"或"relaxed"
        """
        self.kg_graph = kg_graph
        self.max_workers = max_workers
        self.validation_cache = {}  # 缓存验证结果
        self.validation_mode = validation_mode
        
    async def validate_path(self, path: List[str], entities: Optional[List[str]] = None) -> bool:
        """
        验证单条路径的有效性
        
        Args:
            path: 关系路径，如 ['relation1', 'relation2']
            entities: 起始实体列表（可选）
            
        Returns:
            是否有效
        """
        # 检查缓存
        path_key = tuple(path)
        if path_key in self.validation_cache:
            return self.validation_cache[path_key]
            
        # 执行验证
        if self.validation_mode == "strict":
            is_valid = self._validate_path_existence(path, entities)
        else:
            # 宽松模式：允许部分匹配
            is_valid = self._validate_path_relaxed(path, entities)
        
        # 缓存结果
        self.validation_cache[path_key] = is_valid
        
        return is_valid
        
    async def validate_paths_batch(self, paths: List[List[str]], 
                                 entities: Optional[List[str]] = None) -> Dict[int, bool]:
        """
        批量验证路径
        
        Args:
            paths: 路径列表
            entities: 起始实体列表
            
        Returns:
            {路径索引: 是否有效}
        """
        results = {}
        
        # 使用线程池并行验证
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self._validate_path_existence, path, entities): idx
                for idx, path in enumerate(paths)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    is_valid = future.result()
                    results[idx] = is_valid
                except Exception as e:
                    logger.error(f"Error validating path {idx}: {e}")
                    results[idx] = False
                    
        return results
        
    def _validate_path_existence(self, path: List[str], entities: Optional[List[str]] = None) -> bool:
        """
        检查路径是否在KG中存在
        
        使用BFS验证是否存在至少一条满足关系序列的实际路径
        """
        if not path:
            return False
            
        # 如果没有指定起始实体，从所有节点开始尝试
        if not entities:
            # 为了效率，可以采样一部分节点
            sample_size = min(100, len(self.kg_graph.nodes))
            entities = list(self.kg_graph.nodes)[:sample_size]
            
        # 尝试从每个起始实体验证路径
        for entity in entities:
            if entity not in self.kg_graph:
                continue
                
            # 使用BFS验证路径
            result_paths = bfs_with_rule(self.kg_graph, entity, path, max_p=1)
            if result_paths:
                return True
                
        return False
        
    def _validate_path_relaxed(self, path: List[str], entities: Optional[List[str]] = None) -> bool:
        """
        宽松验证模式：允许部分路径匹配或近似匹配
        
        Args:
            path: 关系路径
            entities: 起始实体列表
            
        Returns:
            是否有效
        """
        if not path:
            return False
            
        # 尝试严格验证
        if self._validate_path_existence(path, entities):
            return True
            
        # 如果严格验证失败，尝试验证子路径
        for i in range(len(path)):
            for j in range(i + 1, len(path) + 1):
                if j - i >= 1:  # 至少包含一个关系
                    subpath = path[i:j]
                    if self._validate_path_existence(subpath, entities):
                        return True
                        
        # 尝试近似匹配 - 查找相似关系
        similar_paths = []
        for i, rel in enumerate(path):
            similar_rels = self._find_similar_relations(rel)
            for similar_rel in similar_rels[:3]:  # 只考虑前3个最相似的
                modified_path = path.copy()
                modified_path[i] = similar_rel
                if self._validate_path_existence(modified_path, entities):
                    return True
                    
        return False
        
    def validate_with_constraints(self, path: List[str], constraints: Dict[str, Any]) -> bool:
        """
        带约束条件的路径验证
        
        Args:
            path: 关系路径
            constraints: 约束条件，如 {
                'start_type': 'Person',
                'end_type': 'Movie',
                'max_length': 3
            }
        """
        # 检查路径长度约束
        if 'max_length' in constraints and len(path) > constraints['max_length']:
            return False
            
        # 检查路径模式约束
        if 'forbidden_relations' in constraints:
            forbidden = set(constraints['forbidden_relations'])
            if any(rel in forbidden for rel in path):
                return False
                
        # 检查实体类型约束
        if 'start_type' in constraints or 'end_type' in constraints:
            # 需要更复杂的验证逻辑，检查路径端点的实体类型
            return self._validate_with_type_constraints(path, constraints)
            
        # 基本存在性验证
        return self._validate_path_existence(path)
        
    def _validate_with_type_constraints(self, path: List[str], constraints: Dict[str, Any]) -> bool:
        """验证带类型约束的路径"""
        # 这里需要访问实体类型信息
        # 简化实现：暂时返回基本验证结果
        return self._validate_path_existence(path)
        
    def find_valid_subpaths(self, path: List[str]) -> List[List[str]]:
        """
        找出路径中所有有效的子路径
        
        对于无效路径，尝试找出其有效的子部分
        
        Args:
            path: 原始路径
            
        Returns:
            有效的子路径列表
        """
        valid_subpaths = []
        
        # 尝试所有可能的子路径
        for i in range(len(path)):
            for j in range(i + 1, len(path) + 1):
                if j - i >= 1:  # 至少包含一个关系
                    subpath = path[i:j]
                    if self._validate_path_existence(subpath):
                        valid_subpaths.append(subpath)
                        
                        # 标记子路径长度
                        coverage = len(subpath) / len(path)
                        if coverage > 0.5:  # 如果覆盖了一半以上的原始路径
                            # 记录这是一个高覆盖率的子路径
                            subpath_key = tuple(subpath)
                            if subpath_key not in self.validation_cache:
                                self.validation_cache[subpath_key] = True
                    
        # 按长度排序，优先返回较长的子路径
        valid_subpaths.sort(key=len, reverse=True)
        
        return valid_subpaths
        
    def suggest_corrections(self, invalid_path: List[str]) -> List[List[str]]:
        """
        为无效路径建议修正方案
        
        Args:
            invalid_path: 无效的路径
            
        Returns:
            可能的修正路径列表
        """
        suggestions = []
        
        # 策略1：尝试相似的关系
        for i, rel in enumerate(invalid_path):
            similar_rels = self._find_similar_relations(rel)
            for similar_rel in similar_rels:
                corrected_path = invalid_path[:i] + [similar_rel] + invalid_path[i+1:]
                if self._validate_path_existence(corrected_path):
                    suggestions.append(corrected_path)
                    
        # 策略2：尝试插入中间关系
        for i in range(len(invalid_path) - 1):
            bridge_rels = self._find_bridge_relations(invalid_path[i], invalid_path[i+1])
            for bridge_rel in bridge_rels:
                corrected_path = invalid_path[:i+1] + [bridge_rel] + invalid_path[i+1:]
                if self._validate_path_existence(corrected_path):
                    suggestions.append(corrected_path)
                    
        # 策略3：尝试删除某个关系
        if len(invalid_path) > 1:
            for i in range(len(invalid_path)):
                shortened_path = invalid_path[:i] + invalid_path[i+1:]
                if self._validate_path_existence(shortened_path):
                    suggestions.append(shortened_path)
                    
        return suggestions
        
    def _find_similar_relations(self, relation: str) -> List[str]:
        """查找相似的关系"""
        # 收集所有关系
        all_relations = set()
        for u, v, data in self.kg_graph.edges(data=True):
            if 'relation' in data:
                all_relations.add(data['relation'])
                
        # 简单的相似度计算（可以用更复杂的方法）
        similar = []
        for rel in all_relations:
            if self._compute_similarity(relation, rel) > 0.7:
                similar.append(rel)
                
        return similar[:5]  # 返回前5个最相似的
        
    def _find_bridge_relations(self, rel1: str, rel2: str) -> List[str]:
        """查找可能连接两个关系的桥接关系"""
        # 这需要分析KG的模式
        # 简化实现：返回常见的桥接关系
        return []
        
    def _compute_similarity(self, s1: str, s2: str) -> float:
        """计算字符串相似度"""
        # 简单的Jaccard相似度
        set1 = set(s1.lower().split('_'))
        set2 = set(s2.lower().split('_'))
        
        if not set1 or not set2:
            return 0.0
            
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        return len(intersection) / len(union)
        
    def get_validation_report(self) -> Dict[str, Any]:
        """获取验证报告"""
        return {
            "total_validations": len(self.validation_cache),
            "valid_paths": sum(1 for v in self.validation_cache.values() if v),
            "invalid_paths": sum(1 for v in self.validation_cache.values() if not v),
            "cache_size": len(self.validation_cache)
        }
        
    def clear_cache(self):
        """清空验证缓存"""
        self.validation_cache.clear()

    def repair_path(self, path: List[str]) -> Tuple[List[str], float]:
        """
        尝试修复无效路径
        
        Args:
            path: 无效的路径
            
        Returns:
            (修复后的路径, 修复置信度)
        """
        # 首先尝试找出有效的子路径
        valid_subpaths = self.find_valid_subpaths(path)
        
        if not valid_subpaths:
            # 如果没有有效子路径，尝试修改关系
            corrected_paths = self.suggest_corrections(path)
            if corrected_paths:
                # 返回第一个修正的路径
                return corrected_paths[0], 0.6
            return path, 0.0  # 无法修复
            
        # 如果有超过一半长度的子路径，直接使用它
        for subpath in valid_subpaths:
            if len(subpath) > len(path) / 2:
                confidence = len(subpath) / len(path)
                return subpath, confidence
                
        # 尝试组合多个子路径
        if len(valid_subpaths) >= 2:
            # 简单组合策略：取最长的两个子路径
            sp1, sp2 = valid_subpaths[:2]
            
            # 检查是否可以连接
            if self._can_connect_subpaths(sp1, sp2):
                combined_path = self._combine_subpaths(sp1, sp2)
                return combined_path, 0.7
                
        # 如果只有短的子路径，使用最长的一个
        if valid_subpaths:
            confidence = len(valid_subpaths[0]) / len(path)
            return valid_subpaths[0], confidence
            
        return path, 0.0  # 默认情况下返回原始路径

    def _can_connect_subpaths(self, path1: List[str], path2: List[str]) -> bool:
        """检查两个子路径是否可以连接"""
        # 简单实现：检查在原图中是否存在连接两个路径的边
        # 这需要实际的实体路径，而不仅仅是关系路径
        # 简化版本：如果两个路径没有重叠，认为它们可以连接
        return len(set(path1).intersection(set(path2))) == 0

    def _combine_subpaths(self, path1: List[str], path2: List[str]) -> List[str]:
        """组合两个子路径"""
        # 简单实现：直接拼接
        # 实际应用中可能需要更复杂的逻辑
        return path1 + path2