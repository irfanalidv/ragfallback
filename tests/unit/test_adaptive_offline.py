"""AdaptiveRAGRetriever + QueryVariationsStrategy driven by a scripted fake LLM."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from ragfallback import AdaptiveRAGRetriever, CostTracker
from ragfallback.strategies import QueryVariationsStrategy
from tests.unit.fakes import KeywordEmbeddings, ScriptedLLM


def _store() -> InMemoryVectorStore:
    vs = InMemoryVectorStore(embedding=KeywordEmbeddings())
    vs.add_documents(
        [
            Document(
                page_content="Refunds are issued within 14 days.", metadata={"source": "billing.md"}
            ),
            Document(
                page_content="API tokens expire after 90 days.", metadata={"source": "auth.md"}
            ),
        ]
    )
    return vs


def test_query_variations_parsing_paths() -> None:
    qv = QueryVariationsStrategy(num_variations=2)
    assert qv.generate_queries("q", {}, 1, ScriptedLLM('["a", "b", "c"]')) == ["q", "a", "b"]
    assert qv.generate_queries("q", {"team": "x"}, 2, ScriptedLLM('["a"]')) == ["a"]
    text = 'Sure! Here you go: ["first rewrite", "second rewrite"]'
    assert qv.generate_queries("q", {}, 2, ScriptedLLM(text)) == ["first rewrite", "second rewrite"]
    lines = "a reworded question one\nanother reworded question"
    assert qv.generate_queries("q", {}, 2, ScriptedLLM(lines)) == [
        "a reworded question one",
        "another reworded question",
    ]
    assert qv.generate_queries("q", {}, 2, ScriptedLLM('{"not": "a list"}')) == []

    class Down:
        def invoke(self, _m):
            raise ConnectionError("offline")

    assert qv.generate_queries("q", {}, 1, Down()) == ["q"]  # original survives an LLM outage


def test_low_confidence_answer_triggers_the_rewrite() -> None:
    # Call order: variations, answer #1, confidence #1, answer #2, confidence #2
    llm = ScriptedLLM(
        '["When do refunds arrive?"]',
        '{"answer": "Not found", "source": ""}',
        "0.1",
        '{"answer": "Within 14 days", "source": "billing.md"}',
        "0.95",
    )
    tracker = CostTracker()
    retriever = AdaptiveRAGRetriever(
        vector_store=_store(),
        llm=llm,  # type: ignore[arg-type]
        embedding_model=KeywordEmbeddings(),
        fallback_strategies=[QueryVariationsStrategy(num_variations=1)],
        max_attempts=3,
        min_confidence=0.7,
        cost_tracker=tracker,
        enable_logging=False,
    )
    result = retriever.query_with_fallback("refund timeline?", return_intermediate_steps=True)
    assert result.answer == "Within 14 days"
    assert result.source == "billing.md"
    assert result.attempts == 2
    assert result.intermediate_steps and len(result.intermediate_steps) == 2
    # ScriptedLLM reports 100 prompt + 20 completion tokens per call at gpt-4o-mini prices.
    assert tracker.total_tokens["input"] == 200
    assert result.cost > 0
    stats = retriever.metrics_collector.get_stats()
    assert stats["total_queries"] == 1 and stats["success_rate"] == 1.0


def test_all_attempts_exhausted_returns_best_effort() -> None:
    llm = ScriptedLLM("[]", '{"answer": "Maybe 14 days", "source": "billing.md"}', "0.4")
    retriever = AdaptiveRAGRetriever(
        vector_store=_store(),
        llm=llm,  # type: ignore[arg-type]
        embedding_model=KeywordEmbeddings(),
        fallback_strategies=[QueryVariationsStrategy(num_variations=1)],
        max_attempts=1,
        min_confidence=0.9,
        enable_logging=False,
    )
    result = retriever.query_with_fallback("refunds?")
    assert result.answer == "Maybe 14 days"
    assert result.confidence == 0.4
    assert retriever.metrics_collector.get_stats()["success_rate"] == 0.0


async def test_async_path_matches_sync() -> None:
    llm = ScriptedLLM("[]", '{"answer": "90 days", "source": "auth.md"}', "0.9")
    retriever = AdaptiveRAGRetriever(
        vector_store=_store(),
        llm=llm,  # type: ignore[arg-type]
        embedding_model=KeywordEmbeddings(),
        fallback_strategies=[QueryVariationsStrategy(num_variations=1)],
        min_confidence=0.5,
        enable_logging=False,
    )
    result = await retriever.aquery_with_fallback("When do API tokens expire?")
    assert result.answer == "90 days" and result.attempts == 1
