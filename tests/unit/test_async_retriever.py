"""Unit tests for AdaptiveRAGRetriever.aquery_with_fallback."""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ragfallback.core.adaptive_retriever import AdaptiveRAGRetriever, QueryResult


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _fake_doc(text: str = "Some context text.", doc_id: str = "doc1"):
    doc = MagicMock()
    doc.page_content = text
    doc.metadata = {"id": doc_id, "source": "test"}
    return doc


def _make_retriever_instance(min_confidence: float = 0.0):
    """Build an AdaptiveRAGRetriever with mocked dependencies."""
    # Vector store
    vs = MagicMock()
    r = MagicMock()
    r.invoke.return_value = [_fake_doc()]
    r.ainvoke = AsyncMock(return_value=[_fake_doc()])
    vs.as_retriever.return_value = r

    # LLM
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(
        content='{"answer": "The answer.", "source": "doc1"}',
        response_metadata={},
    )
    llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content='{"answer": "The answer.", "source": "doc1"}',
            response_metadata={},
        )
    )

    # Embeddings
    emb = MagicMock()

    retriever = AdaptiveRAGRetriever(
        vector_store=vs,
        llm=llm,
        embedding_model=emb,
        max_attempts=2,
        min_confidence=min_confidence,
        enable_logging=False,
    )
    return retriever


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestAqueryWithFallbackIsAsync:
    """Interface contract."""

    def test_is_coroutine_function(self):
        """aquery_with_fallback must be an async def."""
        assert inspect.iscoroutinefunction(
            AdaptiveRAGRetriever.aquery_with_fallback
        )

    def test_coexists_with_sync_method(self):
        """Both sync and async methods exist on the same instance."""
        r = _make_retriever_instance()
        assert callable(r.query_with_fallback)
        assert inspect.iscoroutinefunction(r.aquery_with_fallback)


@pytest.mark.asyncio
class TestAqueryWithFallbackReturns:
    """Return-value correctness."""

    async def test_returns_query_result(self):
        """aquery_with_fallback returns a QueryResult."""
        retriever = _make_retriever_instance(min_confidence=0.0)
        with patch.object(
            retriever.metrics_collector, "record_success"
        ), patch.object(
            retriever.metrics_collector, "record_failure"
        ):
            result = await retriever.aquery_with_fallback("What is the policy?")
        assert isinstance(result, QueryResult)
        assert isinstance(result.answer, str)
        assert isinstance(result.confidence, float)
        assert result.attempts >= 1

    async def test_records_success_metrics(self):
        """record_success is called when confidence threshold is met."""
        retriever = _make_retriever_instance(min_confidence=0.0)
        with patch.object(
            retriever.metrics_collector, "record_success"
        ) as mock_success, patch.object(
            retriever.metrics_collector, "record_failure"
        ):
            await retriever.aquery_with_fallback("What is the return window?")
        mock_success.assert_called_once()

    async def test_records_failure_metrics_when_low_confidence(self):
        """record_failure is called when confidence stays below threshold."""
        retriever = _make_retriever_instance(min_confidence=1.0)  # impossible to pass
        # Patch ConfidenceScorer to always return 0
        with patch(
            "ragfallback.core.adaptive_retriever.ConfidenceScorer"
        ) as MockScorer, patch.object(
            retriever.metrics_collector, "record_failure"
        ) as mock_fail, patch.object(
            retriever.metrics_collector, "record_success"
        ):
            MockScorer.return_value.score.return_value = 0.0
            await retriever.aquery_with_fallback("impossible question")
        mock_fail.assert_called_once()


@pytest.mark.asyncio
class TestAqueryFallbackToThreadPool:
    """Graceful degradation when ainvoke is absent."""

    async def test_falls_back_when_ainvoke_missing(self):
        """If LLM raises AttributeError on ainvoke, we fall back to sync in executor."""
        retriever = _make_retriever_instance(min_confidence=0.0)

        # Remove ainvoke from the LLM so the AttributeError path is taken
        del retriever.llm.ainvoke

        with patch.object(
            retriever.metrics_collector, "record_success"
        ), patch.object(
            retriever.metrics_collector, "record_failure"
        ):
            result = await retriever.aquery_with_fallback("fallback test question")

        assert isinstance(result, QueryResult)

    async def test_result_structure_consistent_with_sync(self):
        """Async result has the same shape as sync result."""
        retriever = _make_retriever_instance(min_confidence=0.0)
        question = "How does billing work?"
        with patch.object(
            retriever.metrics_collector, "record_success"
        ), patch.object(
            retriever.metrics_collector, "record_failure"
        ):
            async_result = await retriever.aquery_with_fallback(question)
            sync_result = retriever.query_with_fallback(question)

        # Both must return valid QueryResult with same field types
        for result in (async_result, sync_result):
            assert isinstance(result, QueryResult)
            assert isinstance(result.answer, str)
            assert isinstance(result.confidence, float)
            assert isinstance(result.attempts, int)
            assert isinstance(result.cost, float)
