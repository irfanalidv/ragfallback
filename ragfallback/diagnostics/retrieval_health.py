"""Lightweight retrieval checks against labeled queries (recall@k style)."""

from __future__ import annotations

import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.vectorstores import VectorStore


def _retrieve_docs(vector_store: VectorStore, query: str, k: int) -> list[Any]:
    r = vector_store.as_retriever(search_kwargs={"k": k})
    invoke = getattr(r, "invoke", None)
    if invoke:
        out = invoke(query)
        return out if isinstance(out, list) else []
    return r.get_relevant_documents(query)


def _doc_key(doc: Any) -> str:
    md = getattr(doc, "metadata", None) or {}
    for k in ("id", "doc_id", "source", "chunk_id"):
        if k in md and md[k] is not None:
            return f"{k}:{md[k]}"
    content = getattr(doc, "page_content", str(doc)) or ""
    return f"hash:{hash(content)}"


@dataclass
class RetrievalHealthReport:
    ok: bool
    cases_run: int
    mean_recall_at_k: float
    per_query: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    avg_latency_ms: float | None = None

    @property
    def hit_rate(self) -> float:
        """Alias for mean recall / substring hit rate in [0, 1]."""
        return self.mean_recall_at_k

    def summary(self) -> str:
        k0 = self.per_query[0].get("k") if self.per_query else None
        k_str = str(k0) if k0 is not None else "?"
        return (
            f"retrieval_health ok={self.ok} cases={self.cases_run} "
            f"mean_recall@{k_str}={self.mean_recall_at_k:.3f}"
        )


class RetrievalHealthCheck:
    """
    Run a small labeled harness: for each query, retrieve top-k and measure whether any
    gold document keys appear. Helps catch scale / routing / embedder regressions early.

    Gold keys should match :func:`_doc_key` logic (metadata ``id``, ``doc_id``, ``source``, …).
    """

    def __init__(self, k: int = 5):
        self.k = k

    def recall_at_k(
        self,
        retrieved: Sequence[Any],
        gold_keys: set[str],
        k: int | None = None,
    ) -> float:
        k = k if k is not None else self.k
        if not gold_keys:
            return 0.0
        top = retrieved[:k]
        found = sum(1 for d in top if _doc_key(d) in gold_keys)
        return 1.0 if found else 0.0

    def run(
        self,
        vector_store: VectorStore,
        cases: Sequence[dict[str, Any]],
        k: int | None = None,
        min_mean_recall: float = 0.8,
    ) -> RetrievalHealthReport:
        """
        Each case: ``{"query": str, "gold_keys": Set[str]}`` where gold_keys match indexed docs.

        Example:
            cases = [{"query": "refund policy", "gold_keys": {"source:billing.md"}}]
        """
        k = k if k is not None else self.k
        notes: list[str] = []
        per_query: list[dict[str, Any]] = []
        scores: list[float] = []

        for i, case in enumerate(cases):
            q = case.get("query") or case.get("q")
            gold = case.get("gold_keys") or case.get("relevant_keys")
            if not q or not gold:
                notes.append(f"case[{i}] skipped: need query and gold_keys")
                continue
            if not isinstance(gold, set):
                gold = set(gold)
            try:
                docs = _retrieve_docs(vector_store, q, k)
            except Exception as e:  # pragma: no cover - integration surface
                notes.append(f"case[{i}] retrieval error: {e}")
                scores.append(0.0)
                per_query.append({"query": q, "k": k, "recall": 0.0, "error": str(e)})
                continue
            r = self.recall_at_k(docs, gold, k=k)
            scores.append(r)
            per_query.append({"query": q, "k": k, "recall": r, "n_retrieved": len(docs)})

        mean = sum(scores) / len(scores) if scores else 0.0
        ok = bool(scores) and mean >= min_mean_recall
        return RetrievalHealthReport(
            ok=ok,
            cases_run=len(scores),
            mean_recall_at_k=mean,
            per_query=per_query,
            notes=notes,
        )

    def run_substring_probes(
        self,
        vector_store: VectorStore,
        probes: dict[str, str],
        k: int | None = None,
        min_mean_hit_rate: float = 0.8,
        case_sensitive: bool = False,
    ) -> RetrievalHealthReport:
        """
        Quick health check: for each ``query → substring``, pass if any retrieved chunk's
        ``page_content`` contains the substring. Use for smoke tests without gold doc IDs.
        """
        k = k if k is not None else self.k
        notes: list[str] = []
        per_query: list[dict[str, Any]] = []
        scores: list[float] = []

        latencies_ms: list[float] = []
        for q, needle in probes.items():
            if not q or not needle:
                notes.append(f"skipped empty probe for query={q!r}")
                continue
            try:
                t0 = time.perf_counter()
                docs = _retrieve_docs(vector_store, q, k)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            except Exception as e:
                notes.append(f"probe {q!r} retrieval error: {e}")
                scores.append(0.0)
                per_query.append({"query": q, "k": k, "recall": 0.0, "error": str(e)})
                continue
            n_norm = needle if case_sensitive else needle.lower()
            hit = False
            for d in docs:
                text = getattr(d, "page_content", "") or ""
                t = text if case_sensitive else text.lower()
                if n_norm in t:
                    hit = True
                    break
            r = 1.0 if hit else 0.0
            scores.append(r)
            per_query.append({"query": q, "k": k, "recall": r, "substring": needle})

        mean = sum(scores) / len(scores) if scores else 0.0
        ok = bool(scores) and mean >= min_mean_hit_rate
        avg_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else None
        return RetrievalHealthReport(
            ok=ok,
            cases_run=len(scores),
            mean_recall_at_k=mean,
            per_query=per_query,
            notes=notes,
            avg_latency_ms=avg_ms,
        )

    def quick_check(
        self,
        vector_store: VectorStore,
        documents: Sequence[Any],
        sample_size: int = 10,
        k: int | None = None,
        min_mean_hit_rate: float = 0.5,
        seed: int | None = None,
    ) -> RetrievalHealthReport:
        """
        Substring smoke probes from random document snippets (no manual gold labels).
        """
        docs = [d for d in documents if (getattr(d, "page_content", "") or "").strip()]
        if not docs:
            return RetrievalHealthReport(
                ok=False,
                cases_run=0,
                mean_recall_at_k=0.0,
                notes=["no documents with page_content"],
            )
        rng = random.Random(seed)
        rng.shuffle(docs)
        sample = docs[: min(sample_size, len(docs))]
        probes: dict[str, str] = {}
        for d in sample:
            text = (getattr(d, "page_content", "") or "").strip()
            if len(text) < 50:
                continue
            snippet = text[15:55]
            q = (text[: min(120, len(text))].split(".")[0] or text[:40]).strip()
            if len(snippet) >= 8 and q:
                probes[q] = snippet
        if not probes:
            return RetrievalHealthReport(
                ok=False,
                cases_run=0,
                mean_recall_at_k=0.0,
                notes=["could not build probes from samples"],
            )
        return self.run_substring_probes(
            vector_store,
            probes,
            k=k,
            min_mean_hit_rate=min_mean_hit_rate,
        )
