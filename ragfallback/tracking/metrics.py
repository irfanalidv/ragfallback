"""Metrics collection for RAG operations."""

from typing import Dict, List, Optional
from dataclasses import dataclass
import logging


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    success: bool
    attempts: int
    confidence: float
    cost: float
    latency_ms: float
    strategy_used: str


class MetricsCollector:
    """
    Collect performance metrics for RAG operations.
    
    Tracks success rates, confidence scores, costs, and latencies.
    """
    
    def __init__(self):
        """Initialize MetricsCollector."""
        self.metrics: List[QueryMetrics] = []
        self.logger = logging.getLogger(__name__)
    
    def record_success(
        self,
        attempts: int,
        confidence: float,
        cost: float,
        latency_ms: Optional[float] = None,
        strategy_used: str = "unknown"
    ):
        """Record a successful query."""
        metric = QueryMetrics(
            success=True,
            attempts=attempts,
            confidence=confidence,
            cost=cost,
            latency_ms=latency_ms or 0.0,
            strategy_used=strategy_used
        )
        self.metrics.append(metric)
    
    def record_failure(
        self,
        attempts: int,
        cost: float,
        latency_ms: Optional[float] = None,
        strategy_used: str = "unknown"
    ):
        """Record a failed query."""
        metric = QueryMetrics(
            success=False,
            attempts=attempts,
            confidence=0.0,
            cost=cost,
            latency_ms=latency_ms or 0.0,
            strategy_used=strategy_used
        )
        self.metrics.append(metric)
    
    def get_stats(self) -> Dict:
        """Get aggregated statistics."""
        if not self.metrics:
            return {
                "total_queries": 0,
                "success_rate": 0.0,
                "avg_confidence": 0.0,
                "avg_attempts": 0.0,
                "avg_latency_ms": 0.0,
                "total_cost": 0.0
            }
        
        total_queries = len(self.metrics)
        successful = [m for m in self.metrics if m.success]
        success_rate = len(successful) / total_queries if total_queries > 0 else 0.0
        
        avg_confidence = (
            sum(m.confidence for m in successful) / len(successful)
            if successful else 0.0
        )
        avg_attempts = sum(m.attempts for m in self.metrics) / total_queries
        avg_latency = (
            sum(m.latency_ms for m in self.metrics) / total_queries
            if any(m.latency_ms > 0 for m in self.metrics) else 0.0
        )
        total_cost = sum(m.cost for m in self.metrics)
        
        return {
            "total_queries": total_queries,
            "success_rate": success_rate,
            "avg_confidence": avg_confidence,
            "avg_attempts": avg_attempts,
            "avg_latency_ms": avg_latency,
            "total_cost": total_cost
        }
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = []

