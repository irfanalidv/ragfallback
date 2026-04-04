"""Golden-set runner: drive AdaptiveRAGRetriever and aggregate Ragas + latency metrics."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Union

import numpy as np

from ragfallback.core.adaptive_retriever import AdaptiveRAGRetriever, QueryResult
from ragfallback.evaluation import recall_at_k
from ragfallback.mlops.ragas_hook import RagasHook, RagasReport

logger = logging.getLogger(__name__)


@dataclass
class GoldenReport:
    """End-to-end golden-set evaluation summary."""

    ragas: RagasReport
    recall_at_3: float
    recall_at_5: float
    latency_p95_ms: float
    latency_mean_ms: float
    fallback_rate: float
    n_samples: int
    timestamp: datetime
    per_sample: List[Dict[str, Any]]
    cache_stats: Optional[Dict[str, Any]] = field(default=None)


class GoldenRunner:
    """Execute each golden item through the adaptive retriever and score with RagasHook."""

    def __init__(
        self,
        retriever: AdaptiveRAGRetriever,
        ragas_hook: RagasHook,
        dataset: Union[str, List[Dict[str, Any]]],
        cache_monitor: Optional[Any] = None,
    ) -> None:
        """Load JSON path or use in-memory list; optionally wrap retriever with CacheMonitor.

        Args:
            retriever: :class:`~ragfallback.core.adaptive_retriever.AdaptiveRAGRetriever`
                instance to query.
            ragas_hook: :class:`~ragfallback.mlops.ragas_hook.RagasHook` for scoring.
            dataset: JSON file path or list of ``{"query", "ground_truth", ...}`` dicts.
            cache_monitor: Optional :class:`~ragfallback.tracking.cache_monitor.CacheMonitor`
                instance. When provided, the retriever's vector store retriever is wrapped
                to track cache hits/misses. Stats appear in ``GoldenReport.cache_stats``.
        """
        self.retriever = retriever
        self.ragas_hook = ragas_hook
        self._cache_monitor = cache_monitor
        if isinstance(dataset, str):
            raw = Path(dataset).read_text(encoding="utf-8")
            self._dataset = json.loads(raw)
        else:
            self._dataset = list(dataset)
        if not self._dataset:
            raise ValueError("Golden dataset is empty")
        if not isinstance(self._dataset, list):
            raise ValueError("Golden dataset must be a JSON array or list of dicts")

    def _retrieve_docs(self, query: str, k: int = 5) -> List[Any]:
        """Fetch top-``k`` documents for context and id extraction."""
        r = self.retriever.vector_store.as_retriever(search_kwargs={"k": k})
        if self._cache_monitor is not None:
            r = self._cache_monitor.wrap_retriever(r, k=k)
        invoke = getattr(r, "invoke", None)
        if invoke is not None:
            return list(invoke(query) or [])
        return list(r.get_relevant_documents(query))

    async def _aretrieve_docs(self, query: str, k: int = 5) -> List[Any]:
        """Async fetch of top-``k`` documents."""
        r = self.retriever.vector_store.as_retriever(search_kwargs={"k": k})
        if self._cache_monitor is not None:
            r = self._cache_monitor.wrap_retriever(r, k=k)
        ainvoke = getattr(r, "ainvoke", None)
        if ainvoke is not None:
            return list(await ainvoke(query) or [])
        loop = asyncio.get_event_loop()
        invoke = getattr(r, "invoke", None)
        if invoke is not None:
            return list(await loop.run_in_executor(None, functools.partial(invoke, query)) or [])
        return []

    def _doc_ids(self, docs: Sequence[Any]) -> List[str]:
        """Stable string ids from document metadata or content hash."""
        out: List[str] = []
        for d in docs:
            md = getattr(d, "metadata", None) or {}
            rid = md.get("id") or md.get("doc_id") or md.get("source")
            if rid is not None:
                out.append(str(rid))
            else:
                pc = getattr(d, "page_content", str(d)) or ""
                out.append(str(hash(pc)))
        return out

    def _contexts_from_docs(self, docs: Sequence[Any]) -> List[str]:
        """Plain-text contexts for evaluation."""
        return [(getattr(d, "page_content", str(d)) or "") for d in docs]

    def _ids_from_intermediate(self, result: QueryResult) -> List[str]:
        """Best-effort doc id list from intermediate steps (often empty)."""
        steps = result.intermediate_steps or []
        for step in reversed(steps):
            if not isinstance(step, dict):
                continue
            if "doc_ids" in step:
                return [str(x) for x in step["doc_ids"]]
        return []

    def _run_single(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Run one golden row synchronously and return diagnostics."""
        query = item["query"]
        gt = item.get("ground_truth", "")
        rel_ids: Set[str] = set(str(x) for x in item.get("relevant_doc_ids", []))

        t0 = time.perf_counter()
        result = self.retriever.query_with_fallback(
            query, return_intermediate_steps=True
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        docs = self._retrieve_docs(query, k=5)
        retrieved_ids = self._ids_from_intermediate(result)
        if not retrieved_ids:
            retrieved_ids = self._doc_ids(docs)
        contexts = self._contexts_from_docs(docs)

        r3 = recall_at_k(retrieved_ids, rel_ids, 3)
        r5 = recall_at_k(retrieved_ids, rel_ids, 5)

        return {
            "question": query,
            "ground_truth": gt,
            "answer": result.answer,
            "confidence": result.confidence,
            "contexts": contexts,
            "latency_ms": latency_ms,
            "retrieved_ids": retrieved_ids,
            "fallback_triggered": result.attempts > 1,
            "recall_at_3": r3,
            "recall_at_5": r5,
        }

    async def _arun_single(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Run one golden row natively async using :meth:`aquery_with_fallback`."""
        query = item["query"]
        gt = item.get("ground_truth", "")
        rel_ids: Set[str] = set(str(x) for x in item.get("relevant_doc_ids", []))

        try:
            t0 = time.perf_counter()
            result = await self.retriever.aquery_with_fallback(
                query, return_intermediate_steps=True
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
        except AttributeError:
            logger.warning(
                "retriever does not support aquery_with_fallback — "
                "falling back to thread pool for query: %s",
                query[:80],
            )
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._run_single, item)

        docs = await self._aretrieve_docs(query, k=5)
        retrieved_ids = self._ids_from_intermediate(result)
        if not retrieved_ids:
            retrieved_ids = self._doc_ids(docs)
        contexts = self._contexts_from_docs(docs)

        r3 = recall_at_k(retrieved_ids, rel_ids, 3)
        r5 = recall_at_k(retrieved_ids, rel_ids, 5)

        return {
            "question": query,
            "ground_truth": gt,
            "answer": result.answer,
            "confidence": result.confidence,
            "contexts": contexts,
            "latency_ms": latency_ms,
            "retrieved_ids": retrieved_ids,
            "fallback_triggered": result.attempts > 1,
            "recall_at_3": r3,
            "recall_at_5": r5,
        }

    def _build_report(
        self,
        per_sample: List[Dict[str, Any]],
        ragas_report: RagasReport,
    ) -> GoldenReport:
        """Aggregate latencies and recall means into a GoldenReport."""
        latencies = [float(s["latency_ms"]) for s in per_sample]
        n = len(per_sample) or 1
        mean_r3 = sum(s["recall_at_3"] for s in per_sample) / n
        mean_r5 = sum(s["recall_at_5"] for s in per_sample) / n
        fb = sum(1 for s in per_sample if s.get("fallback_triggered")) / n
        p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
        mean_lat = sum(latencies) / len(latencies) if latencies else 0.0
        cache_stats = (
            self._cache_monitor.get_stats().as_dict()
            if self._cache_monitor is not None
            else None
        )
        return GoldenReport(
            ragas=ragas_report,
            recall_at_3=mean_r3,
            recall_at_5=mean_r5,
            latency_p95_ms=p95,
            latency_mean_ms=mean_lat,
            fallback_rate=fb,
            n_samples=len(per_sample),
            timestamp=datetime.utcnow(),
            per_sample=per_sample,
            cache_stats=cache_stats,
        )

    def run(self) -> GoldenReport:
        """Synchronously evaluate the full golden set."""
        per_sample: List[Dict[str, Any]] = []
        for item in self._dataset:
            per_sample.append(self._run_single(item))
        ragas_samples = [
            {
                "question": s["question"],
                "answer": s["answer"],
                "contexts": s["contexts"],
                "ground_truth": s["ground_truth"],
            }
            for s in per_sample
        ]
        ragas_rep = self.ragas_hook.evaluate_sync(ragas_samples)
        return self._build_report(per_sample, ragas_rep)

    async def run_async(self) -> GoldenReport:
        """Evaluate golden rows concurrently using native async, then Ragas async."""
        tasks = [self._arun_single(item) for item in self._dataset]
        per_sample = list(await asyncio.gather(*tasks))
        ragas_samples = [
            {
                "question": s["question"],
                "answer": s["answer"],
                "contexts": s["contexts"],
                "ground_truth": s["ground_truth"],
            }
            for s in per_sample
        ]
        ragas_rep = await self.ragas_hook.evaluate_async(ragas_samples)
        return self._build_report(per_sample, ragas_rep)
