"""
USE CASE: UC-5 — Hybrid dense + BM25 retrieval with failover
=============================================================
Real problem : Dense retrieval fails on exact keyword queries like "RFC 2822 email
               format" — BM25 recovers them. Additionally, primary vector stores can
               go down or return 0 results for edge-case queries, with no fallback.
Goal         : ``SmartThresholdHybridRetriever`` falls back to BM25 when vector scores
               are weak; ``FailoverRetriever`` switches to a backup store automatically
               when primary returns < min_results docs or raises.
Module       : ragfallback.retrieval.SmartThresholdHybridRetriever
               ragfallback.retrieval.FailoverRetriever
Vector DB    : FAISS (local, no server) + BM25 over same corpus
Install      : pip install ragfallback[hybrid,huggingface]
Env vars     : NONE required
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from a repository clone without pip install -e .
_repo_root = Path(__file__).resolve().parent.parent
if (_repo_root / "ragfallback").is_dir() and str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import _kb_common  # noqa: E402
from ragfallback.retrieval import FailoverRetriever, SmartThresholdHybridRetriever  # noqa: E402


def demo_hybrid_bm25_fallback(docs, vs) -> None:
    """Demonstrate SmartThresholdHybridRetriever falling back to BM25."""
    print("\nSmart threshold + BM25 fallback")

    # max_distance=1e-9 means no vector result passes the L2 threshold,
    # forcing BM25 every time — makes the fallback path clearly visible.
    hybrid = SmartThresholdHybridRetriever.from_documents(
        docs,
        vectorstore=vs,
        fetch_k=12,
        final_k=4,
        score_mode="distance",
        max_distance=1e-9,
    )

    query = "rate limit"
    print(f"\nQuery: '{query}'")
    print("(vector threshold is deliberately strict → BM25 path will activate)\n")

    docs_out = hybrid.invoke(query)
    print(f"Results: {len(docs_out)} doc(s) retrieved")
    for i, d in enumerate(docs_out, 1):
        source = d.metadata.get("ragfallback_retrieval_source", "unknown")
        preview = d.page_content[:80].replace("\n", " ")
        print(f"  [{i}] source={source!r}  text='{preview}…'")

    # _dense_scores_are_weak is part of the public API — demonstrate it
    print(f"\n  _dense_scores_are_weak([])  → {hybrid._dense_scores_are_weak([])}")
    print(f"  _dense_scores_are_weak([doc]) → {hybrid._dense_scores_are_weak(docs_out[:1])}")


def demo_failover_on_empty(vs) -> None:
    """Demonstrate FailoverRetriever when primary returns 0 results."""
    print("\nFailoverRetriever — fallback on empty primary")

    class EmptyRetriever:
        """Simulates a retriever that always returns nothing (e.g. index gap)."""
        def invoke(self, query: str):
            return []

    backup = vs.as_retriever(search_kwargs={"k": 2})
    fb = FailoverRetriever(primary=EmptyRetriever(), fallback=backup, min_results=1)

    query = "billing refund"
    print(f"\nQuery: '{query}'  (primary always returns 0 docs)")
    docs_out = fb.invoke(query)
    print(f"Results: {len(docs_out)} doc(s) from fallback")
    for d in docs_out:
        print(f"  source={d.metadata.get('ragfallback_retrieval_source')!r}  "
              f"text='{d.page_content[:60].replace(chr(10), ' ')}…'")


def demo_failover_on_exception(vs) -> None:
    """Demonstrate FailoverRetriever when primary raises."""
    print("\nFailoverRetriever — fallback on primary exception")

    class BrokenRetriever:
        """Simulates a downed remote vector store."""
        def invoke(self, query: str):
            raise ConnectionError("Pinecone endpoint unreachable")

    backup = vs.as_retriever(search_kwargs={"k": 2})
    fb = FailoverRetriever(primary=BrokenRetriever(), fallback=backup, log_errors=True)

    query = "API authentication"
    print(f"\nQuery: '{query}'  (primary raises ConnectionError)")
    docs_out = fb.invoke(query)
    print(f"Results: {len(docs_out)} doc(s) from fallback")
    for d in docs_out:
        print(f"  source={d.metadata.get('ragfallback_retrieval_source')!r}  "
              f"text='{d.page_content[:60].replace(chr(10), ' ')}…'")


def demo_failover_min_results(vs) -> None:
    """Demonstrate FailoverRetriever with min_results > 1."""
    print("\nFailoverRetriever — min_results=3 triggers fallback on thin results")

    # Primary returns only 1 doc for this query (simulate by limiting k=1)
    thin_primary = vs.as_retriever(search_kwargs={"k": 1})
    rich_backup = vs.as_retriever(search_kwargs={"k": 4})

    fb = FailoverRetriever(primary=thin_primary, fallback=rich_backup, min_results=3)

    query = "refund policy"
    print(f"\nQuery: '{query}'  (primary limited to k=1; min_results=3 → fallback)")
    docs_out = fb.invoke(query)
    print(f"Results: {len(docs_out)} doc(s)  (fallback returned more)")
    for d in docs_out:
        print(f"  source={d.metadata.get('ragfallback_retrieval_source')!r}  "
              f"text='{d.page_content[:60].replace(chr(10), ' ')}…'")


def main() -> None:
    print("UC-5: Hybrid dense+BM25 retrieval with automatic failover")

    docs = _kb_common.load_sample_kb()
    emb = _kb_common.get_embeddings()
    vs = _kb_common.build_faiss_store(docs, emb)

    print(f"\nKnowledge base: {len(docs)} chunks loaded into FAISS.\n")

    demo_hybrid_bm25_fallback(docs, vs)
    demo_failover_on_empty(vs)
    demo_failover_on_exception(vs)
    demo_failover_min_results(vs)

    print("\nDone.")


if __name__ == "__main__":
    main()
