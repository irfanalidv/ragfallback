"""Golden-set runner: drive AdaptiveRAGRetriever and aggregate Ragas + latency metrics."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Union

import numpy as np

from ragfallback.core.adaptive_retriever import AdaptiveRAGRetriever, QueryResult
from ragfallback.evaluation import recall_at_k
from ragfallback.mlops.ragas_hook import RagasHook, RagasReport


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


class GoldenRunner:
    """Execute each golden item through the adaptive retriever and score with RagasHook."""

    def __init__(
        self,
        retriever: AdaptiveRAGRetriever,
        ragas_hook: RagasHook,
        dataset: Union[str, List[Dict[str, Any]]],
    ) -> None:
        """Load JSON path or use in-memory list of ``query`` / ``ground_truth`` / optional ids."""
        self.retriever = retriever
        self.ragas_hook = ragas_hook
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
        invoke = getattr(r, "invoke", None)
        if invoke is not None:
            return list(invoke(query) or [])
        return list(r.get_relevant_documents(query))

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
        return [
            (getattr(d, "page_content", str(d)) or "") for d in docs
        ]

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
        """Run one golden row and return diagnostics plus ragas-oriented fields."""
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
        fallback_triggered = result.attempts > 1

        return {
            "question": query,
            "ground_truth": gt,
            "answer": result.answer,
            "confidence": result.confidence,
            "contexts": contexts,
            "latency_ms": latency_ms,
            "retrieved_ids": retrieved_ids,
            "fallback_triggered": fallback_triggered,
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
        """Evaluate golden rows concurrently (thread pool per row), then Ragas async."""
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(None, self._run_single, item)
            for item in self._dataset
        ]
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
