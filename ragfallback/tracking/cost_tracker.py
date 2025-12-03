"""Cost tracking for RAG operations."""

from typing import Dict, Optional
from dataclasses import dataclass
from contextlib import contextmanager
import logging


@dataclass
class ModelPricing:
    """Pricing configuration for a model."""
    model: str
    input_cost_per_1M: float
    output_cost_per_1M: float


class CostTracker:
    """
    Track token costs across RAG operations.
    
    Supports multiple models and provides budget enforcement.
    """
    
    # Default pricing (per 1M tokens)
    DEFAULT_PRICING = {
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
        budget: Optional[float] = None,
        pricing_config: Optional[Dict[str, ModelPricing]] = None,
        alert_threshold: float = 0.8
    ):
        """
        Initialize CostTracker.
        
        Args:
            budget: Maximum budget in USD
            pricing_config: Custom pricing configuration
            alert_threshold: Alert when budget usage exceeds this fraction
        """
        self.budget = budget
        self.pricing_config = pricing_config or {}
        self.alert_threshold = alert_threshold
        self.total_cost = 0.0
        self.total_tokens = {"input": 0, "output": 0}
        self.operation_costs = {}
        self.logger = logging.getLogger(__name__)
        
        # Track current operation
        self._current_operation = None
        self._current_tokens = {"input": 0, "output": 0}
        self._last_operation_cost = 0.0
    
    @contextmanager
    def track(self, operation: str = "default"):
        """Context manager to track costs for an operation."""
        self._current_operation = operation
        self._current_tokens = {"input": 0, "output": 0}
        try:
            yield self
        finally:
            cost = self._calculate_cost(self._current_tokens)
            self.total_cost += cost
            self.total_tokens["input"] += self._current_tokens["input"]
            self.total_tokens["output"] += self._current_tokens["output"]
            self.operation_costs[operation] = self.operation_costs.get(operation, 0.0) + cost
            self._last_operation_cost = cost
            self._current_operation = None
            
            # Check budget alert
            if self.budget and self.total_cost >= self.budget * self.alert_threshold:
                self.logger.warning(
                    f"Budget alert: {self.total_cost:.4f} / {self.budget:.4f} "
                    f"({self.total_cost / self.budget * 100:.1f}%)"
                )
    
    def record_tokens(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "gpt-4"
    ):
        """Record token usage."""
        if self._current_operation:
            self._current_tokens["input"] += input_tokens
            self._current_tokens["output"] += output_tokens
    
    def _calculate_cost(self, tokens: Dict[str, int], model: str = "gpt-4") -> float:
        """Calculate cost from tokens."""
        # Try custom pricing first, then default
        pricing = self.pricing_config.get(model) or self.DEFAULT_PRICING.get(model)
        if not pricing:
            self.logger.warning(f"No pricing found for model {model}, using default")
            pricing = self.DEFAULT_PRICING.get("gpt-4")
        
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
    
    def get_report(self) -> Dict:
        """Get detailed cost report."""
        return {
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "breakdown": self.operation_costs,
            "budget_remaining": (self.budget - self.total_cost) if self.budget else None,
            "budget_usage_percent": (self.total_cost / self.budget * 100) if self.budget else None
        }

