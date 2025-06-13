# src/got_reasoning/minimal_got.py
"""
最小化GoT实现
只保留核心功能，确保不损害F1分数
"""

from typing import List, Tuple, Dict, Any
import numpy as np


class MinimalGoT:
    """最小化GoT：只做安全的增强"""
    
    def __init__(self, preserve_original: bool = True):
        self.preserve_original = preserve_original
        self.stats = {"calls": 0, "enhanced": 0}
        
    def enhance(self, question: str, paths: List[List[str]]) -> Tuple[List[List[str]], Dict]:
        """
        最小化增强：只做最安全的操作
        """
        self.stats["calls"] += 1
        
        if not paths:
            return paths, {"action": "none", "reason": "empty input"}
            
        # 原始路径
        result = paths[:] if self.preserve_original else []
        
        # 只做一种增强：路径去重和排序
        unique_paths = []
        seen = set()
        
        for path in paths:
            path_tuple = tuple(path)
            if path_tuple not in seen:
                seen.add(path_tuple)
                unique_paths.append(path)
                
        # 按长度排序（短路径优先）
        unique_paths.sort(key=len)
        
        # 只保留前5条
        result = unique_paths[:5]
        
        # 简单的路径扩展（只对最短路径）
        if len(result) > 0 and len(result[0]) == 1:
            # 单跳路径可能不完整，尝试扩展
            extended = self._safe_extend(result[0])
            if extended and extended not in result:
                result.append(extended)
                self.stats["enhanced"] += 1
                
        return result, {
            "action": "minimal_enhancement",
            "original_count": len(paths),
            "final_count": len(result),
            "enhanced": self.stats["enhanced"]
        }
        
    def _safe_extend(self, path: List[str]) -> List[str]:
        """安全的路径扩展"""
        # 只对特定模式做扩展
        if len(path) == 1:
            rel = path[0]
            
            # 常见的需要扩展的模式
            extensions = {
                "location.location.contains": ["location.location.contains"],
                "person.person.nationality": ["location.country.official_language"],
                "organization.organization.headquarters": ["location.mailing_address.citytown"]
            }
            
            if rel in extensions:
                return path + extensions[rel]
                
        return None
        
        
def integrate_minimal_got(question: str, paths: List[List[str]]) -> List[List[str]]:
    """
    集成函数：一行代码调用
    """
    got = MinimalGoT(preserve_original=True)
    enhanced_paths, _ = got.enhance(question, paths)
    return enhanced_paths