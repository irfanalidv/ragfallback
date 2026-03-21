"""
ragfallback - RAG Fallback Strategies Library

Prefer subpackage imports in application code, e.g.:
``from ragfallback.diagnostics import ChunkQualityChecker``.

This module exposes a small curated shortcut only (see ``__all__``).
"""

from __future__ import annotations

__version__ = "2.0.1"
__author__ = "Irfan Ali"

from ragfallback.core.adaptive_retriever import AdaptiveRAGRetriever, QueryResult
from ragfallback.tracking.cost_tracker import CostTracker
from ragfallback.tracking.metrics import MetricsCollector

__all__ = [
    "AdaptiveRAGRetriever",
    "QueryResult",
    "CostTracker",
    "MetricsCollector",
]
