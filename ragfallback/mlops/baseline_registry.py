"""JSON baseline storage and regression gating for golden-set evaluation."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_LATENCY_KEY = "latency_p95_ms"


class RegressionError(Exception):
    """Raised when one or more metrics regress against a stored baseline."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class BaselineRegistry:
    """Load/save metric baselines per dataset name and compare new runs."""

    def __init__(self, path: str = "baselines.json") -> None:
        """Set storage path and load existing JSON if present."""
        self.path = Path(path)
        self._data: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Populate ``_data`` from disk or start empty."""
        if not self.path.exists():
            self._data = {}
            return
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not read baselines from %s — starting empty.", self.path)
            self._data = {}

    def _save(self) -> None:
        """Persist ``_data`` with indentation."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def exists(self, dataset: str) -> bool:
        """Return True if a baseline exists for ``dataset``."""
        return dataset in self._data

    def get(self, dataset: str) -> Dict[str, Any]:
        """Return the baseline dict for ``dataset`` or raise KeyError."""
        if dataset not in self._data:
            raise KeyError(f"No baseline registered for dataset '{dataset}'")
        return dict(self._data[dataset])

    def update(self, report: "GoldenReport", dataset: str) -> None:
        """Persist aggregated metrics from ``report`` under ``dataset``."""
        from ragfallback.mlops.golden_runner import GoldenReport

        if not isinstance(report, GoldenReport):
            raise TypeError("report must be a GoldenReport")
        r = report.ragas
        entry = {
            "faithfulness": float(r.faithfulness),
            "answer_relevance": float(r.answer_relevance),
            "context_precision": float(r.context_precision),
            "context_recall": float(r.context_recall),
            "recall_at_3": float(report.recall_at_3),
            "recall_at_5": float(report.recall_at_5),
            "latency_p95_ms": float(report.latency_p95_ms),
            "recorded_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        }
        self._data[dataset] = entry
        self._save()

    def compare_or_fail(
        self,
        report: "GoldenReport",
        dataset: str,
        threshold: float = 0.05,
        latency_threshold: Optional[float] = None,
    ) -> None:
        """Raise RegressionError if any metric regresses beyond ``threshold``.

        If ``latency_threshold`` is set, it applies only to ``latency_p95_ms``
        (higher allows more runner noise); otherwise latency uses ``threshold``.
        """
        from ragfallback.mlops.golden_runner import GoldenReport

        if not isinstance(report, GoldenReport):
            raise TypeError("report must be a GoldenReport")
        lat_t = latency_threshold if latency_threshold is not None else threshold
        if dataset not in self._data:
            logger.warning(
                "No baseline for '%s' — skipping regression check.", dataset
            )
            return
        baseline = self.get(dataset)
        lines: List[str] = []
        r = report.ragas

        def check_score(name: str, new_val: float, old_val: Any) -> None:
            try:
                old_f = float(old_val)
            except (TypeError, ValueError):
                return
            if new_val < old_f * (1.0 - threshold):
                lines.append(
                    f"  {name}: {old_f:.4f} → {new_val:.4f} (Δ{new_val - old_f:+.4f})"
                )

        check_score("faithfulness", r.faithfulness, baseline.get("faithfulness"))
        check_score(
            "answer_relevance", r.answer_relevance, baseline.get("answer_relevance")
        )
        check_score(
            "context_precision",
            r.context_precision,
            baseline.get("context_precision"),
        )
        check_score(
            "context_recall", r.context_recall, baseline.get("context_recall")
        )
        check_score(
            "recall_at_3", report.recall_at_3, baseline.get("recall_at_3")
        )
        check_score(
            "recall_at_5", report.recall_at_5, baseline.get("recall_at_5")
        )

        try:
            new_lat = float(report.latency_p95_ms)
            old_lat = float(baseline.get(_LATENCY_KEY, new_lat))
            if new_lat > old_lat * (1.0 + lat_t):
                lines.append(
                    f"  {_LATENCY_KEY}: {old_lat:.4f} → {new_lat:.4f} "
                    f"(Δ{new_lat - old_lat:+.4f})"
                )
        except (TypeError, ValueError):
            pass

        if lines:
            msg = "Regression detected for '{}':\n{}".format(dataset, "\n".join(lines))
            raise RegressionError(msg)
