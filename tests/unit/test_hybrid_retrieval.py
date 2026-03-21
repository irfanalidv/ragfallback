"""Unit tests for SmartThresholdHybridRetriever and FailoverRetriever (Session 4 spec)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from ragfallback.retrieval.failover import FailoverRetriever
from ragfallback.retrieval.smart_hybrid import SmartThresholdHybridRetriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(text: str) -> Document:
    return Document(page_content=text)


def _make_dense_store(pairs: list[tuple[str, float]]) -> MagicMock:
    """Return a mock vector store whose similarity_search_with_score returns *pairs*."""
    store = MagicMock()
    store.similarity_search_with_score.return_value = [
        (_doc(text), score) for text, score in pairs
    ]
    return store


def _make_bm25_retriever(texts: list[str]) -> MagicMock:
    """Return a mock BM25 retriever that returns documents via get_relevant_documents."""
    r = MagicMock(spec=["get_relevant_documents"])
    r.get_relevant_documents.return_value = [_doc(t) for t in texts]
    return r


def _make_retriever_with_invoke(texts: list[str]) -> MagicMock:
    r = MagicMock()
    r.invoke.return_value = [_doc(t) for t in texts]
    return r


# ---------------------------------------------------------------------------
# SmartThresholdHybridRetriever — _dense_scores_are_weak
# ---------------------------------------------------------------------------


def test_dense_scores_are_weak_on_empty_list() -> None:
    r = SmartThresholdHybridRetriever(
        vectorstore=_make_dense_store([]),
        score_mode="distance",
        max_distance=0.5,
    )
    assert r._dense_scores_are_weak([]) is True


def test_dense_scores_are_weak_on_non_empty_list() -> None:
    r = SmartThresholdHybridRetriever(
        vectorstore=_make_dense_store([]),
        score_mode="distance",
        max_distance=0.5,
    )
    assert r._dense_scores_are_weak([_doc("something")]) is False


# ---------------------------------------------------------------------------
# SmartThresholdHybridRetriever — BM25 fallback when dense is weak
# ---------------------------------------------------------------------------


def test_bm25_fallback_when_dense_threshold_filters_all() -> None:
    """When all dense results are too far, BM25 should fill in."""
    store = _make_dense_store([("irrelevant", 99.0)])  # score >> max_distance
    bm25 = _make_bm25_retriever(["BM25 keyword hit"])
    r = SmartThresholdHybridRetriever(
        vectorstore=store,
        bm25_retriever=bm25,
        score_mode="distance",
        max_distance=0.5,
        fetch_k=5,
        final_k=3,
    )
    docs = r.invoke("exact keyword query")
    assert len(docs) == 1
    assert docs[0].page_content == "BM25 keyword hit"
    assert docs[0].metadata["ragfallback_retrieval_source"] == "bm25_fallback"


def test_bm25_not_called_when_dense_returns_results() -> None:
    """BM25 must be skipped when dense retrieval succeeds."""
    store = _make_dense_store([("dense hit", 0.1)])  # score <= max_distance
    bm25 = _make_bm25_retriever(["BM25 result"])
    r = SmartThresholdHybridRetriever(
        vectorstore=store,
        bm25_retriever=bm25,
        score_mode="distance",
        max_distance=0.5,
        fetch_k=5,
        final_k=3,
    )
    docs = r.invoke("query")
    assert len(docs) == 1
    assert docs[0].page_content == "dense hit"
    assert docs[0].metadata["ragfallback_retrieval_source"] == "vector_threshold"
    bm25.get_relevant_documents.assert_not_called()


def test_safe_empty_when_no_bm25_and_dense_fails() -> None:
    """No BM25, dense filtered empty → return []."""
    store = _make_dense_store([("far", 99.0)])
    r = SmartThresholdHybridRetriever(
        vectorstore=store,
        bm25_retriever=None,
        score_mode="distance",
        max_distance=0.5,
        fetch_k=5,
        final_k=3,
    )
    assert r.invoke("q") == []


# ---------------------------------------------------------------------------
# SmartThresholdHybridRetriever — BM25 ImportError is actionable
# ---------------------------------------------------------------------------


def test_from_documents_raises_import_error_without_rank_bm25() -> None:
    """from_documents must raise ImportError with pip install hint if rank_bm25 absent."""
    store = _make_dense_store([])
    docs = [_doc("doc")]
    with patch(
        "ragfallback.retrieval.smart_hybrid.logger"
    ):  # silence logging
        with patch(
            "langchain_community.retrievers.BM25Retriever.from_documents",
            side_effect=ImportError("no module named rank_bm25"),
        ):
            with pytest.raises(ImportError, match="pip install ragfallback\\[hybrid\\]"):
                SmartThresholdHybridRetriever.from_documents(docs, vectorstore=store)


# ---------------------------------------------------------------------------
# SmartThresholdHybridRetriever — duck-typing (get_relevant_documents)
# ---------------------------------------------------------------------------


def test_smart_hybrid_exposes_invoke() -> None:
    """SmartThresholdHybridRetriever must expose invoke() as the public retrieval API."""
    store = _make_dense_store([("result", 0.1)])
    r = SmartThresholdHybridRetriever(
        vectorstore=store,
        score_mode="distance",
        max_distance=0.5,
        fetch_k=5,
        final_k=3,
    )
    assert callable(getattr(r, "invoke", None))
    docs = r.invoke("test query")
    assert isinstance(docs, list)


# ---------------------------------------------------------------------------
# FailoverRetriever — fallback when primary returns 0 results
# ---------------------------------------------------------------------------


def test_failover_uses_fallback_when_primary_returns_empty() -> None:
    """When primary returns 0 docs (< min_results=1), fallback must be used."""
    primary = _make_retriever_with_invoke([])
    fallback = _make_retriever_with_invoke(["fallback doc"])
    fb = FailoverRetriever(primary=primary, fallback=fallback, min_results=1)
    docs = fb.invoke("query")
    assert len(docs) == 1
    assert docs[0].page_content == "fallback doc"
    assert docs[0].metadata["ragfallback_retrieval_source"] == "secondary_failover"


def test_failover_uses_fallback_when_primary_raises() -> None:
    """When primary raises an exception, fallback must be used."""
    primary = MagicMock()
    primary.invoke.side_effect = ConnectionError("host unreachable")
    fallback = _make_retriever_with_invoke(["backup result"])
    fb = FailoverRetriever(primary=primary, fallback=fallback, log_errors=False)
    docs = fb.invoke("query")
    assert docs[0].page_content == "backup result"
    assert docs[0].metadata["ragfallback_retrieval_source"] == "secondary_failover"


def test_failover_uses_primary_when_results_meet_min() -> None:
    """Primary with ≥ min_results docs must NOT trigger fallback."""
    primary = _make_retriever_with_invoke(["primary result"])
    fallback = MagicMock()  # must not be called
    fb = FailoverRetriever(primary=primary, fallback=fallback, min_results=1)
    docs = fb.invoke("query")
    assert docs[0].metadata["ragfallback_retrieval_source"] == "primary"
    fallback.invoke.assert_not_called()


def test_failover_min_results_threshold() -> None:
    """min_results=2: primary returning 1 doc should trigger fallback."""
    primary = _make_retriever_with_invoke(["only one"])
    fallback = _make_retriever_with_invoke(["fallback a", "fallback b"])
    fb = FailoverRetriever(primary=primary, fallback=fallback, min_results=2)
    docs = fb.invoke("query")
    assert all(d.metadata["ragfallback_retrieval_source"] == "secondary_failover" for d in docs)


# ---------------------------------------------------------------------------
# FailoverRetriever — backward compat: secondary= still works
# ---------------------------------------------------------------------------


def test_failover_secondary_alias_still_works() -> None:
    """Legacy secondary= parameter must still function."""
    primary = _make_retriever_with_invoke([])
    secondary = _make_retriever_with_invoke(["legacy backup"])
    fb = FailoverRetriever(primary=primary, secondary=secondary, log_errors=False)
    docs = fb.invoke("query")
    assert docs[0].page_content == "legacy backup"


def test_failover_requires_fallback_or_secondary() -> None:
    """Constructing without fallback or secondary should raise ValueError."""
    with pytest.raises((ValueError, Exception)):
        FailoverRetriever(primary=MagicMock())


# ---------------------------------------------------------------------------
# FailoverRetriever — duck-typing (get_relevant_documents)
# ---------------------------------------------------------------------------


def test_failover_exposes_invoke() -> None:
    """FailoverRetriever must expose invoke() as the public retrieval API."""
    primary = _make_retriever_with_invoke(["doc"])
    fallback = _make_retriever_with_invoke([])
    fb = FailoverRetriever(primary=primary, fallback=fallback)
    assert callable(getattr(fb, "invoke", None))
    docs = fb.invoke("query")
    assert isinstance(docs, list)


# ---------------------------------------------------------------------------
# FailoverRetriever — returns empty safely when fallback also fails
# ---------------------------------------------------------------------------


def test_failover_returns_empty_when_both_fail() -> None:
    """If both primary and fallback raise, return [] instead of propagating."""
    primary = MagicMock()
    primary.invoke.side_effect = RuntimeError("primary down")
    fallback = MagicMock()
    fallback.invoke.side_effect = RuntimeError("fallback down too")
    fb = FailoverRetriever(
        primary=primary, fallback=fallback, log_errors=False
    )
    docs = fb.invoke("query")
    assert docs == []
