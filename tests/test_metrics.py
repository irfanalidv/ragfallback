"""Tests for MetricsCollector."""

import pytest
from ragfallback.tracking.metrics import MetricsCollector


def test_metrics_collector_initialization():
    """Test MetricsCollector initialization."""
    metrics = MetricsCollector()
    assert len(metrics.metrics) == 0


def test_metrics_collector_record_success():
    """Test recording successful query."""
    metrics = MetricsCollector()
    
    metrics.record_success(
        attempts=2,
        confidence=0.85,
        cost=0.05,
        latency_ms=500.0,
        strategy_used="query_variations"
    )
    
    assert len(metrics.metrics) == 1
    assert metrics.metrics[0].success is True
    assert metrics.metrics[0].confidence == 0.85


def test_metrics_collector_record_failure():
    """Test recording failed query."""
    metrics = MetricsCollector()
    
    metrics.record_failure(
        attempts=3,
        cost=0.10,
        latency_ms=1000.0,
        strategy_used="query_variations"
    )
    
    assert len(metrics.metrics) == 1
    assert metrics.metrics[0].success is False
    assert metrics.metrics[0].confidence == 0.0


def test_metrics_collector_get_stats():
    """Test getting aggregated statistics."""
    metrics = MetricsCollector()
    
    metrics.record_success(attempts=1, confidence=0.9, cost=0.05)
    metrics.record_success(attempts=2, confidence=0.8, cost=0.08)
    metrics.record_failure(attempts=3, cost=0.10)
    
    stats = metrics.get_stats()
    assert stats["total_queries"] == 3
    assert stats["success_rate"] == pytest.approx(2/3, rel=0.01)
    assert stats["avg_confidence"] == pytest.approx(0.85, rel=0.01)
    assert stats["total_cost"] == pytest.approx(0.23, rel=0.01)


def test_metrics_collector_reset():
    """Test resetting metrics."""
    metrics = MetricsCollector()
    
    metrics.record_success(attempts=1, confidence=0.9, cost=0.05)
    assert len(metrics.metrics) == 1
    
    metrics.reset()
    assert len(metrics.metrics) == 0

