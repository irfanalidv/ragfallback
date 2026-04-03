"""
ragfallback.mlops — MLOps-grade evaluation, regression gating, and load testing.

Install extras: pip install ragfallback[mlops]
"""

from ragfallback.mlops.ragas_hook import RagasHook, RagasReport
from ragfallback.mlops.baseline_registry import BaselineRegistry, RegressionError
from ragfallback.mlops.golden_runner import GoldenRunner, GoldenReport
from ragfallback.mlops.query_simulator import QuerySimulator, SimQuery
from ragfallback.mlops.mlflow_logger import MLflowLogger
from ragfallback.mlops.locust_template import generate_locustfile

__all__ = [
    "RagasHook",
    "RagasReport",
    "BaselineRegistry",
    "RegressionError",
    "GoldenRunner",
    "GoldenReport",
    "QuerySimulator",
    "SimQuery",
    "MLflowLogger",
    "generate_locustfile",
]
