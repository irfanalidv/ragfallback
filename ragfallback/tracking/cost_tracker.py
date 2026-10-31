"""Cost tracking for RAG operations."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass
class ModelPricing:
    """Pricing configuration for a model."""

    model: str
    input_cost_per_1M: float
    output_cost_per_1M: float


class CostTracker:
    """Token cost ledger for a RAG session.

    Records spend per operation, enforces an optional budget ceiling, and
    surfaces a report after the session. Pricing defaults cover the main
    OpenAI and Anthropic model families; pass pricing_config to override.
    """

    # Default pricing (per 1M tokens)
    DEFAULT_PRICING: ClassVar[dict[str, ModelPricing]] = {
        "gpt-4": ModelPricing("gpt-4", 30.0, 60.0),
        "gpt-4-turbo": ModelPricing("gpt-4-turbo", 10.0, 30.0),
        "gpt-3.5-turbo": ModelPricing("gpt-3.5-turbo", 0.5, 1.5),
        "gpt-4o": ModelPricing("gpt-4o", 5.0, 15.0),
        "gpt-4o-mini": ModelPricing("gpt-4o-mini", 0.15, 0.6),
        "claude-3-opus": ModelPricing("claude-3-opus", 15.0, 75.0),
        "claude-3-sonnet": ModelPricing("claude-3-sonnet", 3.0, 15.0),
        "claude-3-haiku": ModelPricing("claude-3-haiku", 0.25, 1.25),
    }

    def __init__(
        self,
        budget: float | None = None,
        pricing_config: dict[str, ModelPricing] | None = None,
        alert_threshold: float = 0.8,
    ) -> None:
        self.budget = budget
        self.pricing_config: dict[str, ModelPricing] = pricing_config or {}
        self.alert_threshold = alert_threshold
        self.total_cost = 0.0
        self.total_tokens: dict[str, int] = {"input": 0, "output": 0}
        self.operation_costs: dict[str, float] = {}
        self.logger = logging.getLogger(__name__)

        self._current_operation: str | None = None
        self._current_tokens: dict[str, int] = {"input": 0, "output": 0}
        self._current_cost = 0.0
        self._last_operation_cost = 0.0

    @contextmanager
    def track(self, operation: str = "default") -> Iterator["CostTracker"]:
        """Context manager that attributes tokens recorded inside it to ``operation``."""
        self._current_operation = operation
        self._current_tokens = {"input": 0, "output": 0}
        self._current_cost = 0.0
        try:
            yield self
        finally:
            self._commit(operation, self._current_tokens, self._current_cost)
            self._current_operation = None

    def record_tokens(self, input_tokens: int, output_tokens: int, model: str = "gpt-4") -> None:
        """Record token usage for ``model``.

        Inside :meth:`track` the tokens are attributed to the active operation and
        committed when the block exits. Outside it they are committed immediately
        under the ``"untracked"`` operation (earlier versions silently dropped them).
        """
        cost = self._calculate_cost({"input": input_tokens, "output": output_tokens}, model)
        if self._current_operation:
            self._current_tokens["input"] += input_tokens
            self._current_tokens["output"] += output_tokens
            self._current_cost += cost
        else:
            self._commit("untracked", {"input": input_tokens, "output": output_tokens}, cost)

    def _commit(self, operation: str, tokens: dict[str, int], cost: float) -> None:
        self.total_cost += cost
        self.total_tokens["input"] += tokens["input"]
        self.total_tokens["output"] += tokens["output"]
        self.operation_costs[operation] = self.operation_costs.get(operation, 0.0) + cost
        self._last_operation_cost = cost

        if self.budget and self.total_cost >= self.budget * self.alert_threshold:
            self.logger.warning(
                "Budget alert: %.4f / %.4f (%.1f%%)",
                self.total_cost,
                self.budget,
                self.total_cost / self.budget * 100,
            )

    def _pricing_for(self, model: str) -> ModelPricing | None:
        """Exact match first, then the longest known prefix (``gpt-4o-mini-2024-07-18``)."""
        table = {**self.DEFAULT_PRICING, **self.pricing_config}
        if model in table:
            return table[model]
        prefixes = [k for k in table if model.startswith(k)]
        return table[max(prefixes, key=len)] if prefixes else None

    def _calculate_cost(self, tokens: dict[str, int], model: str = "gpt-4") -> float:
        """Calculate USD cost of ``tokens`` at ``model``'s per-1M-token price."""
        pricing = self._pricing_for(model)
        if pricing is None:
            self.logger.warning("no pricing for model %r, using gpt-4 pricing", model)
            pricing = self.DEFAULT_PRICING["gpt-4"]
        input_cost = (tokens["input"] / 1_000_000) * pricing.input_cost_per_1M
        output_cost = (tokens["output"] / 1_000_000) * pricing.output_cost_per_1M
        return input_cost + output_cost

    def get_last_cost(self) -> float:
        """Get cost of last tracked operation."""
        return self._last_operation_cost

    def budget_exceeded(self) -> bool:
        """Check if budget has been exceeded."""
        if self.budget is None:
            return False
        return self.total_cost >= self.budget

    def get_report(self) -> dict[str, Any]:
        """Get detailed cost report."""
        return {
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "breakdown": self.operation_costs,
            "budget_remaining": ((self.budget - self.total_cost) if self.budget else None),
            "budget_usage_percent": (
                (self.total_cost / self.budget * 100) if self.budget else None
            ),
        }
