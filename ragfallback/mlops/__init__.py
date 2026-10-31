"""
ragfallback.mlops — MLOps-grade evaluation, regression gating, and load testing.

Install extras: pip install ragfallback[mlops]
"""

from ragfallback.mlops.baseline_registry import BaselineRegistry, RegressionError
from ragfallback.mlops.golden_runner import GoldenReport, GoldenRunner
from ragfallback.mlops.locust_template import generate_locustfile
from ragfallback.mlops.mlflow_logger import MLflowLogger
from ragfallback.mlops.query_simulator import QuerySimulator, SimQuery
from ragfallback.mlops.ragas_hook import RagasHook, RagasReport

__all__ = [
    "BaselineRegistry",
    "GoldenReport",
    "GoldenRunner",
    "MLflowLogger",
    "QuerySimulator",
    "RagasHook",
    "RagasReport",
    "RegressionError",
    "SimQuery",
    "generate_locustfile",
]
