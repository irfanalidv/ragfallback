"""Production retrieval reliability: thresholds, hybrid BM25, failover."""

from ragfallback.retrieval.failover import FailoverRetriever
from ragfallback.retrieval.rerank_guard import ReRankerGuard
from ragfallback.retrieval.smart_hybrid import SmartThresholdHybridRetriever
from ragfallback.retrieval.wrappers import RetrieverAsVectorStore

__all__ = [
    "SmartThresholdHybridRetriever",
    "FailoverRetriever",
    "ReRankerGuard",
    "RetrieverAsVectorStore",
]
