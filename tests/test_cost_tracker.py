"""Tests for CostTracker."""

import pytest
from ragfallback.tracking.cost_tracker import CostTracker, ModelPricing


def test_cost_tracker_initialization():
    """Test CostTracker initialization."""
    tracker = CostTracker()
    assert tracker.total_cost == 0.0
    assert tracker.budget is None


def test_cost_tracker_with_budget():
    """Test CostTracker with budget."""
    tracker = CostTracker(budget=10.0)
    assert tracker.budget == 10.0
    assert not tracker.budget_exceeded()


def test_cost_tracker_record_tokens():
    """Test recording tokens."""
    tracker = CostTracker()
    
    with tracker.track(operation="test"):
        tracker.record_tokens(input_tokens=1000, output_tokens=500, model="gpt-4")
    
    assert tracker.total_tokens["input"] == 1000
    assert tracker.total_tokens["output"] == 500
    assert tracker.total_cost > 0


def test_cost_tracker_budget_exceeded():
    """Test budget exceeded check."""
    tracker = CostTracker(budget=0.01)  # Very small budget
    
    with tracker.track(operation="test"):
        tracker.record_tokens(input_tokens=10000, output_tokens=5000, model="gpt-4")
    
    assert tracker.budget_exceeded()


def test_cost_tracker_get_report():
    """Test cost report generation."""
    tracker = CostTracker(budget=10.0)
    
    with tracker.track(operation="test1"):
        tracker.record_tokens(input_tokens=1000, output_tokens=500, model="gpt-4")
    
    report = tracker.get_report()
    assert "total_cost" in report
    assert "total_tokens" in report
    assert "breakdown" in report
    assert report["budget_remaining"] is not None

