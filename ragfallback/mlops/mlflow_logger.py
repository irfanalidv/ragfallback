"""Optional MLflow logging for golden-set reports."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragfallback.mlops.golden_runner import GoldenReport

logger = logging.getLogger(__name__)


class MLflowLogger:
    """Log :class:`~ragfallback.mlops.golden_runner.GoldenReport` metrics when MLflow is installed."""

    def __init__(self, tracking_uri: str = "file:./mlruns") -> None:
        """Configure tracking URI or become a no-op if MLflow is missing."""
        self._mlflow_available = False
        self._mlflow = None  # type: ignore[assignment]
        try:
            import mlflow

            self._mlflow = mlflow
            self._mlflow_available = True
            mlflow.set_tracking_uri(tracking_uri)
        except ImportError:
            logger.warning(
                "mlflow not installed — MLflowLogger is a no-op. "
                "Install with: pip install ragfallback[mlops]"
            )

    def log_golden_report(
        self,
        report: "GoldenReport",
        run_name: str,
        dataset: str = "unknown",
    ) -> None:
        """Start a run and log params/metrics from ``report``; swallow MLflow errors."""
        from ragfallback.mlops.golden_runner import GoldenReport as GR

        if not isinstance(report, GR):
            raise TypeError("report must be a GoldenReport")
        if not self._mlflow_available or self._mlflow is None:
            logger.debug("MLflow not available, skipping.")
            return
        mlflow = self._mlflow
        try:
            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("dataset", dataset)
                mlflow.log_param("n_samples", report.n_samples)
                mlflow.log_param("timestamp", report.timestamp.isoformat())
                mlflow.log_param("fallback_mode", report.ragas.fallback_mode)
                mlflow.log_metric("faithfulness", float(report.ragas.faithfulness))
                mlflow.log_metric(
                    "answer_relevance", float(report.ragas.answer_relevance)
                )
                mlflow.log_metric(
                    "context_precision", float(report.ragas.context_precision)
                )
                mlflow.log_metric("context_recall", float(report.ragas.context_recall))
                mlflow.log_metric("recall_at_3", float(report.recall_at_3))
                mlflow.log_metric("recall_at_5", float(report.recall_at_5))
                mlflow.log_metric("latency_p95_ms", float(report.latency_p95_ms))
                mlflow.log_metric("latency_mean_ms", float(report.latency_mean_ms))
                mlflow.log_metric("fallback_rate", float(report.fallback_rate))
        except Exception as exc:
            logger.warning("MLflow logging failed: %s", exc)
