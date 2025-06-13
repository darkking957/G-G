# src/got_reasoning/__init__.py
"""
Graph of Thoughts (GoT) Reasoning Module

This module implements the Graph of Thoughts framework for enhancing
reasoning on knowledge graphs.
"""

from .got_engine import GoTEngine, GoTConfig
from .thought_graph import ThoughtGraph, ThoughtNode, ThoughtEdge
from .validate_plans import PlanValidator
from .evaluate_plans import SemanticEvaluator, EvaluationResult
from .aggregate_plans import ThoughtAggregator, AggregationResult
from .graph_attention import (
    GraphAttentionNetwork, 
    ThoughtGraphAttention,
    apply_gat_to_thought_graph
)
from .feedback_loop import FeedbackController, FeedbackSignal, ImprovementStrategy
from .minimal_got import MinimalGoT,integrate_minimal_got

__all__ = [
    # Engine
    "GoTEngine",
    "GoTConfig",
    
    # Graph structures
    "ThoughtGraph",
    "ThoughtNode", 
    "ThoughtEdge",
    
    # Components
    "PlanValidator",
    "SemanticEvaluator",
    "EvaluationResult",
    "ThoughtAggregator",
    "AggregationResult",
    "GraphAttentionNetwork",
    "ThoughtGraphAttention",
    "FeedbackController",
    "FeedbackSignal",
    "ImprovementStrategy",
    
    # Utilities
    "apply_gat_to_thought_graph",
    "integrate_minimal_got",
    "MinimalGoT"
]

__version__ = "1.0.0"
__author__ = "GoT-RoG Team"