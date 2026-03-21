"""Unit tests for the MultiHopFallbackStrategy bridge in AdaptiveRAGRetriever."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from ragfallback import AdaptiveRAGRetriever, QueryResult
from ragfallback.strategies.multi_hop import (
    HopResult,
    MultiHopFallbackStrategy,
    MultiHopResult,
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _make_mock_vector_store(docs: Optional[List] = None):
    from langchain_core.documents import Document

    if docs is None:
        docs = [Document(page_content="Acme Corp was acquired by GlobalTech.", metadata={})]

    vs = MagicMock()
    retriever = MagicMock()
    retriever.get_relevant_documents.return_value = docs
    vs.as_retriever.return_value = retriever
    return vs


def _make_mock_llm():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="Not found")
    return llm


def _make_mock_embeddings():
    emb = MagicMock()
    emb.embed_documents.return_value = [[0.1] * 4]
    emb.embed_query.return_value = [0.1] * 4
    return emb


def _make_successful_hop_result() -> MultiHopResult:
    hop = HopResult(
        hop_number=1,
        sub_question="Who acquired Acme Corp?",
        retrieved_chunks=["Acme Corp was acquired by GlobalTech."],
        partial_answer="GlobalTech",
        confidence=0.9,
    )
    return MultiHopResult(
        final_answer="GlobalTech acquired Acme Corp and had $5B revenue.",
        hops=[hop],
        total_hops=1,
        success=True,
    )


def _make_failed_hop_result() -> MultiHopResult:
    return MultiHopResult(
        final_answer="",
        hops=[],
        total_hops=0,
        success=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMultiHopBridgeSuccess:
    """AdaptiveRAGRetriever correctly delegates to MultiHopFallbackStrategy.run()."""

    def test_result_source_is_multi_hop(self):
        vs = _make_mock_vector_store()
        llm = _make_mock_llm()
        emb = _make_mock_embeddings()
        strategy = MagicMock(spec=MultiHopFallbackStrategy)
        strategy.run.return_value = _make_successful_hop_result()

        retriever = AdaptiveRAGRetriever(
            vector_store=vs,
            llm=llm,
            embedding_model=emb,
            fallback_strategies=[strategy],
        )
        result = retriever.query_with_fallback("What was GlobalTech's revenue after acquiring Acme?")

        assert isinstance(result, QueryResult)
        assert result.source == "multi_hop"

    def test_final_answer_propagated(self):
        vs = _make_mock_vector_store()
        llm = _make_mock_llm()
        emb = _make_mock_embeddings()
        hop_result = _make_successful_hop_result()
        strategy = MagicMock(spec=MultiHopFallbackStrategy)
        strategy.run.return_value = hop_result

        retriever = AdaptiveRAGRetriever(
            vector_store=vs,
            llm=llm,
            embedding_model=emb,
            fallback_strategies=[strategy],
        )
        result = retriever.query_with_fallback("What was GlobalTech's revenue after acquiring Acme?")

        assert result.answer == hop_result.final_answer

    def test_confidence_is_085_on_success(self):
        vs = _make_mock_vector_store()
        llm = _make_mock_llm()
        emb = _make_mock_embeddings()
        strategy = MagicMock(spec=MultiHopFallbackStrategy)
        strategy.run.return_value = _make_successful_hop_result()

        retriever = AdaptiveRAGRetriever(
            vector_store=vs,
            llm=llm,
            embedding_model=emb,
            fallback_strategies=[strategy],
        )
        result = retriever.query_with_fallback("complex multi-hop question")

        assert result.confidence == pytest.approx(0.85)

    def test_run_called_with_question_and_retriever_and_llm(self):
        vs = _make_mock_vector_store()
        llm = _make_mock_llm()
        emb = _make_mock_embeddings()
        strategy = MagicMock(spec=MultiHopFallbackStrategy)
        strategy.run.return_value = _make_successful_hop_result()

        retriever = AdaptiveRAGRetriever(
            vector_store=vs,
            llm=llm,
            embedding_model=emb,
            fallback_strategies=[strategy],
        )
        question = "What is the multi-hop question?"
        retriever.query_with_fallback(question)

        strategy.run.assert_called_once()
        call_kwargs = strategy.run.call_args
        assert call_kwargs.kwargs.get("question") == question or call_kwargs.args[0] == question

    def test_intermediate_steps_include_strategy_key(self):
        vs = _make_mock_vector_store()
        llm = _make_mock_llm()
        emb = _make_mock_embeddings()
        strategy = MagicMock(spec=MultiHopFallbackStrategy)
        strategy.run.return_value = _make_successful_hop_result()

        retriever = AdaptiveRAGRetriever(
            vector_store=vs,
            llm=llm,
            embedding_model=emb,
            fallback_strategies=[strategy],
        )
        result = retriever.query_with_fallback("question", return_intermediate_steps=True)

        assert result.intermediate_steps is not None
        assert len(result.intermediate_steps) >= 1
        step = result.intermediate_steps[0]
        assert step.get("strategy") == "multi_hop"

    def test_intermediate_steps_none_when_not_requested(self):
        vs = _make_mock_vector_store()
        llm = _make_mock_llm()
        emb = _make_mock_embeddings()
        strategy = MagicMock(spec=MultiHopFallbackStrategy)
        strategy.run.return_value = _make_successful_hop_result()

        retriever = AdaptiveRAGRetriever(
            vector_store=vs,
            llm=llm,
            embedding_model=emb,
            fallback_strategies=[strategy],
        )
        result = retriever.query_with_fallback("question", return_intermediate_steps=False)

        assert result.intermediate_steps is None


class TestMultiHopBridgeFailure:
    """When the multi-hop run() fails, fallback proceeds normally."""

    def test_failed_hop_does_not_raise(self):
        vs = _make_mock_vector_store()
        llm = _make_mock_llm()
        emb = _make_mock_embeddings()
        strategy = MagicMock(spec=MultiHopFallbackStrategy)
        strategy.run.return_value = _make_failed_hop_result()

        retriever = AdaptiveRAGRetriever(
            vector_store=vs,
            llm=llm,
            embedding_model=emb,
            fallback_strategies=[strategy],
        )
        result = retriever.query_with_fallback("hard question")

        assert isinstance(result, QueryResult)

    def test_failed_hop_result_has_low_confidence(self):
        vs = _make_mock_vector_store()
        llm = _make_mock_llm()
        emb = _make_mock_embeddings()
        strategy = MagicMock(spec=MultiHopFallbackStrategy)
        strategy.run.return_value = _make_failed_hop_result()

        retriever = AdaptiveRAGRetriever(
            vector_store=vs,
            llm=llm,
            embedding_model=emb,
            fallback_strategies=[strategy],
        )
        result = retriever.query_with_fallback("hard question")

        assert result.confidence < 0.5


class TestNonMultiHopStrategiesUnchanged:
    """Strategies without run() still use generate_queries() path as before."""

    def test_generate_queries_called_for_normal_strategy(self):
        from ragfallback.strategies.base import FallbackStrategy

        class _SimpleFakeStrategy(FallbackStrategy):
            def generate_queries(self, original_query, context, attempt, llm):
                return [original_query]

            def get_name(self):
                return "fake"

        vs = _make_mock_vector_store(docs=[])
        llm = _make_mock_llm()
        emb = _make_mock_embeddings()
        strategy = _SimpleFakeStrategy()

        retriever = AdaptiveRAGRetriever(
            vector_store=vs,
            llm=llm,
            embedding_model=emb,
            fallback_strategies=[strategy],
        )
        result = retriever.query_with_fallback("simple question")

        # Should not error — normal path ran
        assert isinstance(result, QueryResult)
        assert result.source != "multi_hop"
