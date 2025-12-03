"""Fallback strategy implementations."""

from ragfallback.strategies.base import FallbackStrategy
from ragfallback.strategies.query_variations import QueryVariationsStrategy

__all__ = ["FallbackStrategy", "QueryVariationsStrategy"]

