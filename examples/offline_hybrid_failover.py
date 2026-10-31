"""
Keyword queries, a dead primary index, and a tight context window - handled.

What it shows
    1. ``SmartThresholdHybridRetriever``: short keyword / acronym queries ("PKCE",
       "Retry-After") score low against long paragraphs in dense space, so the
       similarity threshold drops them and the retriever falls back to BM25
       instead of returning noise or nothing.
    2. ``FailoverRetriever``: the primary (e.g. a hosted vector DB) raises during an
       outage; the query is served from a local fallback index instead.
    3. ``ContextWindowGuard``: retrieved chunks are ranked and trimmed to a token
       budget before they reach the prompt.

What it needs
    ``pip install "ragfallback[hybrid]"`` (adds rank_bm25). No network, no API key;
    uses the bundled ``examples/sample_kb`` articles and a hashing embedder.

Run
    python examples/offline_hybrid_failover.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _offline import HashingEmbeddings, load_paragraphs
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore

from ragfallback.diagnostics import ContextWindowGuard
from ragfallback.retrieval import FailoverRetriever, SmartThresholdHybridRetriever


class OfflineVectorDB(BaseRetriever):
    """Stands in for a hosted vector database that is currently unreachable."""

    def _get_relevant_documents(self, query: str, *, run_manager: Any) -> list[Document]:
        raise ConnectionError("vector-db.example.com: connection refused")


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="   [log] %(name)s: %(message)s")
    docs = load_paragraphs()
    embeddings = HashingEmbeddings()
    store = InMemoryVectorStore(embedding=embeddings)
    store.add_documents(docs)

    print("1) Threshold + BM25 fallback", flush=True)
    hybrid = SmartThresholdHybridRetriever.from_documents(
        docs,
        vectorstore=store,
        score_mode="similarity",  # InMemoryVectorStore returns cosine similarity
        min_similarity=0.3,
        final_k=2,
    )
    for query in [
        "How are refunds handled for annual plan customers who cancel?",
        "PKCE",
        "Retry-After",
    ]:
        hits = hybrid.invoke(query)
        path = hits[0].metadata["ragfallback_retrieval_source"] if hits else "none"
        top = hits[0].page_content[:70] if hits else "-"
        print(f"   {query[:45]!r:<48} via {path:<16} -> {top}...")

    print("\n2) Failover when the primary index is down", flush=True)
    failover = FailoverRetriever(
        primary=OfflineVectorDB(),
        fallback=store.as_retriever(search_kwargs={"k": 3}),
        min_results=1,
    )
    served = failover.invoke("How many requests per minute do Pro accounts get?")
    print(f"   served {len(served)} docs from the fallback; top: {served[0].page_content[:80]}...")

    print("\n3) Trim to a context budget")
    guard = ContextWindowGuard(max_context_tokens=120)
    selected, report = guard.select(
        "What happens when I exceed the rate limit?", served + hybrid.invoke("429"), embeddings
    )
    print(f"   {report.summary()}")
    for d in selected:
        print(f"     kept [{d.metadata['source']}] {d.page_content[:60]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
