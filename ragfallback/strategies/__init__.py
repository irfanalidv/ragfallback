"""Fallback strategy implementations."""

from ragfallback.strategies.base import FallbackStrategy
from ragfallback.strategies.multi_hop import HopResult, MultiHopFallbackStrategy, MultiHopResult
from ragfallback.strategies.query_variations import QueryVariationsStrategy

__all__ = [
    "FallbackStrategy",
    "HopResult",
    "MultiHopFallbackStrategy",
    "MultiHopResult",
    "QueryVariationsStrategy",
]









