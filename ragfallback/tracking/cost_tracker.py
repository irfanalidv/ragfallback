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


# Standard (non-batch, uncached) list prices in USD per 1M tokens, checked against the
# providers' official pricing pages on PRICING_AS_OF:
#   OpenAI     https://developers.openai.com/api/docs/pricing
#   Anthropic  https://platform.claude.com/docs/en/about-claude/pricing
#   Mistral    https://docs.mistral.ai/inference/pricing
# Models with long-context tiers (OpenAI flagships above 272K tokens, Claude Haiku 5.5 above
# 100K tokens) are listed at their short-context price. Prices change: pass ``pricing_config``
# to override any entry.
PRICING_AS_OF = "2026-10-31"


def _p(model: str, input_per_1m: float, output_per_1m: float) -> tuple[str, ModelPricing]:
    return model, ModelPricing(model, input_per_1m, output_per_1m)


_DEFAULT_PRICING: dict[str, ModelPricing] = dict(
    [
        # OpenAI: current models
        _p("gpt-6-astra", 10.0, 50.0),
        _p("gpt-6.1-sol", 2.0, 10.0),
        _p("gpt-6.1-terra", 1.0, 5.0),
        _p("gpt-6-sol", 2.0, 10.0),
        _p("gpt-6-luna", 0.10, 0.50),
        _p("gpt-5.6-sol", 4.0, 20.0),
        _p("gpt-5.6-terra", 2.0, 12.0),
        _p("gpt-5.6-luna", 0.20, 1.20),
        _p("gpt-5.5", 5.0, 30.0),
        _p("gpt-5.5-pro", 30.0, 180.0),
        _p("gpt-5.4", 2.50, 15.0),
        _p("gpt-5.4-pro", 30.0, 180.0),
        _p("gpt-5.4-mini", 0.75, 4.50),
        _p("gpt-5.4-nano", 0.20, 1.25),
        _p("gpt-5.3-codex", 1.75, 14.0),
        _p("gpt-5.2", 1.75, 14.0),
        _p("gpt-5.2-pro", 21.0, 168.0),
        _p("gpt-5.1", 1.25, 10.0),
        _p("gpt-5", 1.25, 10.0),
        _p("gpt-5-mini", 0.25, 2.0),
        _p("gpt-5-nano", 0.05, 0.40),
        _p("gpt-5-pro", 15.0, 120.0),
        _p("gpt-4.1", 2.0, 8.0),
        _p("gpt-4.1-mini", 0.40, 1.60),
        _p("gpt-4.1-nano", 0.10, 0.40),
        _p("gpt-4o", 2.50, 10.0),
        _p("gpt-4o-2024-05-13", 5.0, 15.0),
        _p("gpt-4o-mini", 0.15, 0.60),
        _p("o1", 15.0, 60.0),
        _p("o1-pro", 150.0, 600.0),
        _p("o3", 2.0, 8.0),
        _p("o3-pro", 20.0, 80.0),
        _p("o3-mini", 1.10, 4.40),
        _p("o4-mini", 1.10, 4.40),
        # OpenAI: legacy models
        _p("gpt-4-turbo", 10.0, 30.0),
        _p("gpt-4", 30.0, 60.0),
        _p("gpt-3.5-turbo", 0.50, 1.50),
        # Anthropic (API ids such as ``claude-opus-5-5``; dated ids match by prefix)
        _p("claude-fable-5-1", 10.0, 50.0),
        _p("claude-fable-5", 10.0, 50.0),
        _p("claude-opus-5-5", 4.0, 20.0),
        _p("claude-opus-5", 5.0, 25.0),
        _p("claude-opus-4-8", 5.0, 25.0),
        _p("claude-opus-4-7", 5.0, 25.0),
        _p("claude-opus-4-6", 5.0, 25.0),
        _p("claude-opus-4-5", 5.0, 25.0),
        _p("claude-opus-4-1", 15.0, 75.0),
        _p("claude-opus-4", 15.0, 75.0),
        _p("claude-sonnet-5-5", 2.0, 10.0),
        _p("claude-sonnet-5", 2.0, 10.0),
        _p("claude-sonnet-4-6", 3.0, 15.0),
        _p("claude-sonnet-4-5", 3.0, 15.0),
        _p("claude-sonnet-4", 3.0, 15.0),
        _p("claude-haiku-5-5", 0.10, 0.50),
        _p("claude-haiku-4-5", 1.0, 5.0),
        _p("claude-3-5-haiku", 0.80, 4.0),
        # Retired Claude 3 models, kept at their last list price for old ledgers.
        _p("claude-3-opus", 15.0, 75.0),
        _p("claude-3-sonnet", 3.0, 15.0),
        _p("claude-3-haiku", 0.25, 1.25),
        # Mistral: family prefixes cover ``-latest`` aliases and dated ids, priced at the
        # current generation (Large 4, Small 4, Ministral 3, Codestral 25.08).
        _p("mistral-large", 0.70, 2.10),
        _p("mistral-small", 0.15, 0.60),
        _p("ministral-14b", 0.20, 0.20),
        _p("ministral-8b", 0.15, 0.15),
        _p("ministral-3b", 0.10, 0.10),
        _p("codestral", 0.30, 0.90),
        _p("mistral-embed", 0.10, 0.0),
        _p("codestral-embed", 0.15, 0.0),
        # OpenAI embeddings (input only)
        _p("text-embedding-3-small", 0.02, 0.0),
        _p("text-embedding-3-large", 0.13, 0.0),
        _p("text-embedding-ada-002", 0.10, 0.0),
    ]
)


class CostTracker:
    """Token cost ledger for a RAG session.

    Records spend per operation, enforces an optional budget ceiling, and
    surfaces a report after the session. The built-in table covers current and
    legacy OpenAI, Anthropic and Mistral models (see ``PRICING_AS_OF``); pass
    ``pricing_config`` to override or extend it. Unknown models are priced at
    the ``gpt-4`` rate, which overestimates almost every current model, so
    budgets fail safe.
    """

    # Default pricing (USD per 1M tokens). See PRICING_AS_OF and the sources above.
    DEFAULT_PRICING: ClassVar[dict[str, ModelPricing]] = _DEFAULT_PRICING

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
