"""Unit tests for MultiHopFallbackStrategy."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ragfallback.strategies.multi_hop import (
    HopResult,
    MultiHopFallbackStrategy,
    MultiHopResult,
    _extract_list_from_text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm(*responses: str) -> MagicMock:
    """Return a mock LLM that yields each response string in order."""
    mock = MagicMock()
    mock.invoke.side_effect = [MagicMock(content=r) for r in responses]
    return mock


def _make_retriever(*chunk_lists: list[str]) -> MagicMock:
    """Return a mock retriever whose ``invoke`` cycles through *chunk_lists*."""
    from langchain_core.documents import Document

    call_count = [0]

    def _invoke(query: str) -> list:
        idx = min(call_count[0], len(chunk_lists) - 1)
        call_count[0] += 1
        return [Document(page_content=c) for c in chunk_lists[idx]]

    r = MagicMock()
    r.invoke.side_effect = _invoke
    return r


# ---------------------------------------------------------------------------
# Initialisation / validation
# ---------------------------------------------------------------------------


def test_init_defaults() -> None:
    s = MultiHopFallbackStrategy()
    assert s.max_hops == 3
    assert s.top_k == 4


def test_init_custom() -> None:
    s = MultiHopFallbackStrategy(max_hops=2, top_k=8)
    assert s.max_hops == 2
    assert s.top_k == 8


def test_init_rejects_zero_hops() -> None:
    with pytest.raises(ValueError, match="max_hops must be >= 1"):
        MultiHopFallbackStrategy(max_hops=0)


def test_init_rejects_negative_hops() -> None:
    with pytest.raises(ValueError, match="max_hops must be >= 1"):
        MultiHopFallbackStrategy(max_hops=-1)


# ---------------------------------------------------------------------------
# get_name (inherited from FallbackStrategy)
# ---------------------------------------------------------------------------


def test_get_name() -> None:
    assert MultiHopFallbackStrategy().get_name() == "MultiHopFallbackStrategy"


# ---------------------------------------------------------------------------
# generate_queries — FallbackStrategy contract
# ---------------------------------------------------------------------------


def test_generate_queries_attempt_1_returns_original() -> None:
    """On the first attempt the strategy should let the direct hit run first."""
    strategy = MultiHopFallbackStrategy(max_hops=2)
    llm = MagicMock()  # must NOT be called
    result = strategy.generate_queries("Some question", {}, attempt=1, llm=llm)
    assert result == ["Some question"]
    llm.invoke.assert_not_called()


def test_generate_queries_attempt_2_decomposes() -> None:
    strategy = MultiHopFallbackStrategy(max_hops=2)
    llm = _make_llm('["Who acquired Acme?", "What was their revenue?"]')
    result = strategy.generate_queries("Complex question", {}, attempt=2, llm=llm)
    assert result == ["Who acquired Acme?", "What was their revenue?"]


def test_generate_queries_respects_max_hops() -> None:
    strategy = MultiHopFallbackStrategy(max_hops=2)
    llm = _make_llm('["Q1", "Q2", "Q3", "Q4"]')
    result = strategy.generate_queries("Big question", {}, attempt=2, llm=llm)
    assert len(result) <= 2


def test_generate_queries_falls_back_to_original_on_llm_error() -> None:
    strategy = MultiHopFallbackStrategy(max_hops=2)
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("LLM unavailable")
    result = strategy.generate_queries("Original question", {}, attempt=2, llm=llm)
    assert result == ["Original question"]


def test_generate_queries_falls_back_when_decomposition_empty() -> None:
    strategy = MultiHopFallbackStrategy(max_hops=2)
    llm = _make_llm("[]")  # empty list — no sub-questions
    result = strategy.generate_queries("Something", {}, attempt=2, llm=llm)
    assert result == ["Something"]


# ---------------------------------------------------------------------------
# run() — orchestration
# ---------------------------------------------------------------------------


def test_run_returns_multi_hop_result() -> None:
    strategy = MultiHopFallbackStrategy(max_hops=2, top_k=2)
    llm = _make_llm(
        '["Who acquired Acme?", "What was their revenue?"]',  # decompose
        "TechCorp acquired Acme.",  # hop 1 answer
        "TechCorp had $5B revenue.",  # hop 2 answer
        "TechCorp acquired Acme and had $5B revenue.",  # synthesis
    )
    retriever = _make_retriever(
        ["TechCorp acquired Acme in 2022."],
        ["TechCorp annual revenue was $5 billion."],
    )
    result = strategy.run(
        "Who acquired Acme and what was their revenue?", retriever, llm
    )

    assert isinstance(result, MultiHopResult)
    assert result.total_hops == 2
    assert result.success is True
    assert len(result.hops) == 2
    assert result.hops[0].hop_number == 1
    assert result.hops[1].hop_number == 2
    assert "TechCorp" in result.final_answer


def test_run_marks_not_found_hops() -> None:
    strategy = MultiHopFallbackStrategy(max_hops=1, top_k=2)
    llm = _make_llm(
        '["What is the answer?"]',  # decompose
        "NOT_FOUND",  # hop 1 answer
        "Could not find an answer.",  # synthesis
    )
    retriever = _make_retriever(["Some irrelevant text that is long enough."])
    result = strategy.run("Unanswerable question", retriever, llm)
    assert result.hops[0].partial_answer == "NOT_FOUND"
    assert result.hops[0].confidence == 0.0


def test_run_handles_empty_retrieval_for_all_hops() -> None:
    strategy = MultiHopFallbackStrategy(max_hops=2, top_k=2)
    llm = _make_llm(
        '["Q1", "Q2"]',  # decompose
        "Synthesis fallback.",  # synthesis (hop answers skipped — no chunks)
    )
    retriever = _make_retriever([], [])
    result = strategy.run("Question with no docs", retriever, llm)
    for hop in result.hops:
        assert hop.partial_answer == "NOT_FOUND"
        assert hop.confidence == 0.0


def test_run_returns_failure_when_decomposition_empty() -> None:
    strategy = MultiHopFallbackStrategy(max_hops=2)
    llm = _make_llm("[]")  # decompose returns empty list
    retriever = MagicMock()
    result = strategy.run("Question", retriever, llm)
    assert result.success is False
    assert result.total_hops == 0


def test_run_handles_json_error_in_decompose_via_text_extraction() -> None:
    """When the LLM returns a bulleted list instead of JSON, we still parse it."""
    strategy = MultiHopFallbackStrategy(max_hops=2)
    llm = _make_llm(
        "- What company acquired Acme?\n- What was their revenue?",  # non-JSON
        "TechCorp acquired Acme.",
        "Revenue was five billion.",
        "TechCorp had five billion revenue.",
    )
    retriever = _make_retriever(
        ["TechCorp acquired Acme Corp."],
        ["Revenue five billion dollars."],
    )
    result = strategy.run("Multi-part question about Acme", retriever, llm)
    assert isinstance(result, MultiHopResult)
    # text extraction should have found at least one sub-question
    assert result.total_hops >= 1


def test_run_uses_get_relevant_documents_when_invoke_absent() -> None:
    """Strategy must fall back to get_relevant_documents() if invoke is missing."""
    from langchain_core.documents import Document

    strategy = MultiHopFallbackStrategy(max_hops=1, top_k=2)
    llm = _make_llm(
        '["What is X?"]',
        "X is the answer.",
        "X is the answer.",
    )
    retriever = MagicMock(spec=["get_relevant_documents"])
    retriever.get_relevant_documents.return_value = [
        Document(page_content="X is definitely the answer here.")
    ]
    result = strategy.run("What is X?", retriever, llm)
    assert result.total_hops == 1
    retriever.get_relevant_documents.assert_called_once()


# ---------------------------------------------------------------------------
# MultiHopResult.summary()
# ---------------------------------------------------------------------------


def test_summary_success() -> None:
    result = MultiHopResult(final_answer="42", hops=[], total_hops=2, success=True)
    s = result.summary()
    assert "SUCCESS" in s
    assert "hops=2" in s
    assert "42" in s


def test_summary_failed() -> None:
    result = MultiHopResult(final_answer="", hops=[], total_hops=0, success=False)
    assert "FAILED" in result.summary()


def test_summary_truncates_long_answer() -> None:
    long_answer = "a" * 200
    result = MultiHopResult(
        final_answer=long_answer, hops=[], total_hops=1, success=True
    )
    assert "…" in result.summary()


# ---------------------------------------------------------------------------
# HopResult dataclass
# ---------------------------------------------------------------------------


def test_hop_result_fields() -> None:
    hop = HopResult(
        hop_number=1,
        sub_question="Q?",
        retrieved_chunks=["chunk1", "chunk2"],
        partial_answer="Answer.",
        confidence=0.8,
    )
    assert hop.hop_number == 1
    assert hop.retrieved_chunks == ["chunk1", "chunk2"]
    assert hop.confidence == 0.8


# ---------------------------------------------------------------------------
# _extract_list_from_text (module-level helper)
# ---------------------------------------------------------------------------


def test_extract_list_from_bracketed_json() -> None:
    text = 'Some preamble ["What is X?", "What is Y?"] trailing text'
    result = _extract_list_from_text(text)
    assert result == ["What is X?", "What is Y?"]


def test_extract_list_from_bulleted_lines() -> None:
    text = "- What company acquired Acme?\n- What was their revenue in 2023?"
    result = _extract_list_from_text(text)
    assert len(result) == 2
    assert "What company acquired Acme?" in result


def test_extract_list_ignores_short_lines() -> None:
    text = "- ok\n- What is the annual revenue of TechCorp after the acquisition?"
    result = _extract_list_from_text(text)
    # "ok" is too short (≤ 10 chars), the long one should survive
    assert all(len(r) > 10 for r in result)
