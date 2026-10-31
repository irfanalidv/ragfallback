"""Tests for CostTracker."""

import pytest

from ragfallback.tracking.cost_tracker import CostTracker


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


@pytest.mark.parametrize(
    "model, expected",
    [
        ("gpt-4o", (2.50, 10.0)),
        ("gpt-4o-2024-08-06", (2.50, 10.0)),
        ("gpt-4o-2024-05-13", (5.0, 15.0)),
        ("gpt-4o-mini-2024-07-18", (0.15, 0.60)),
        ("gpt-4.1-mini", (0.40, 1.60)),
        ("gpt-5-mini", (0.25, 2.0)),
        ("gpt-5.4-mini", (0.75, 4.50)),
        ("o3-mini", (1.10, 4.40)),
        ("claude-sonnet-4-5-20250929", (3.0, 15.0)),
        ("claude-opus-4-1-20250805", (15.0, 75.0)),
        ("claude-haiku-4-5", (1.0, 5.0)),
        ("mistral-small-latest", (0.15, 0.60)),
        ("mistral-large-latest", (0.70, 2.10)),
        ("codestral-latest", (0.30, 0.90)),
        ("codestral-embed-2505", (0.15, 0.0)),
    ],
)
def test_builtin_prices_match_provider_list_prices(model, expected):
    pricing = CostTracker()._pricing_for(model)
    assert pricing is not None
    assert (pricing.input_cost_per_1M, pricing.output_cost_per_1M) == expected


def test_pricing_table_is_dated():
    from ragfallback.tracking.cost_tracker import PRICING_AS_OF

    assert len(PRICING_AS_OF) == 10 and PRICING_AS_OF[4] == "-"


def test_llm_model_name_reads_mistral_style_model_attribute():
    from types import SimpleNamespace

    from ragfallback.core.adaptive_retriever import _llm_model_name

    assert _llm_model_name(SimpleNamespace(model_name="gpt-4o-mini")) == "gpt-4o-mini"
    assert _llm_model_name(SimpleNamespace(model="mistral-small-latest")) == "mistral-small-latest"
    assert _llm_model_name(SimpleNamespace()) == "gpt-4"
