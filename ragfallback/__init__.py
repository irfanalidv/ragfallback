"""
ragfallback - RAG Fallback Strategies Library

A production-ready Python library that adds intelligent fallback strategies
to RAG systems, preventing silent failures and improving answer quality.
"""

__version__ = "0.1.0"
__author__ = "Irfan Ali"

from ragfallback.core.adaptive_retriever import AdaptiveRAGRetriever, QueryResult
from ragfallback.strategies.query_variations import QueryVariationsStrategy
from ragfallback.tracking.cost_tracker import CostTracker, ModelPricing
from ragfallback.tracking.metrics import MetricsCollector

__all__ = [
    "AdaptiveRAGRetriever",
    "QueryResult",
    "QueryVariationsStrategy",
    "CostTracker",
    "ModelPricing",
    "MetricsCollector",
]

