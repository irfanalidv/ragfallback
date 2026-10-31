"""Offline tests for factories, env helpers, CostTracker pricing, and small helpers."""

from __future__ import annotations

import sys
import types

import pytest
from langchain_core.documents import Document

from ragfallback._compat import content_hash, invoke_retriever
from ragfallback.evaluation.rag_evaluator import RAGEvaluator, _invoke_llm_text
from ragfallback.retrieval import ReRankerGuard, RetrieverAsVectorStore
from ragfallback.tracking.cost_tracker import CostTracker, ModelPricing
from ragfallback.utils import embedding_factory, env, llm_factory, vector_store_factory
from ragfallback.utils.confidence_scorer import ConfidenceScorer
from tests.unit.fakes import KeywordEmbeddings, ListRetriever, ScriptedLLM

# ── CostTracker ──────────────────────────────────────────────────────────────


def test_cost_tracker_prices_each_model() -> None:
    tracker = CostTracker()
    with tracker.track("answer"):
        tracker.record_tokens(1_000_000, 0, model="gpt-4o-mini")
    assert tracker.total_cost == pytest.approx(0.15)
    assert tracker.get_last_cost() == pytest.approx(0.15)

    with tracker.track("answer"):
        tracker.record_tokens(0, 1_000_000, model="gpt-4o-mini-2024-07-18")  # prefix match
    assert tracker.operation_costs["answer"] == pytest.approx(0.75)


def test_cost_tracker_untracked_custom_pricing_and_unknown_model() -> None:
    tracker = CostTracker(pricing_config={"my-model": ModelPricing("my-model", 1.0, 2.0)})
    tracker.record_tokens(1_000_000, 1_000_000, model="my-model")
    assert tracker.operation_costs == {"untracked": pytest.approx(3.0)}
    assert tracker.total_tokens == {"input": 1_000_000, "output": 1_000_000}

    tracker.record_tokens(1_000_000, 0, model="totally-unknown")  # falls back to gpt-4
    assert tracker.total_cost == pytest.approx(3.0 + 30.0)


def test_cost_tracker_budget_report() -> None:
    tracker = CostTracker(budget=1.0)
    with tracker.track("x"):
        tracker.record_tokens(100_000, 0, model="gpt-4")  # $3.00
    assert tracker.budget_exceeded()
    report = tracker.get_report()
    assert report["budget_remaining"] == pytest.approx(-2.0)
    assert report["budget_usage_percent"] == pytest.approx(300.0)


# ── Confidence heuristic ─────────────────────────────────────────────────────


def test_confidence_heuristic_does_not_zero_answers_containing_x() -> None:
    scorer = ConfidenceScorer(method="heuristic")
    docs = [Document(page_content="d")] * 5
    assert scorer.score("Where?", "The office is in Austin, Texas.", docs) > 0.5
    assert scorer.score("Where?", "x", docs) == 0.0
    assert scorer.score("Where?", "Unknown.", docs) == 0.0
    assert scorer.score("Where?", "The answer was not found in the documents", docs) == 0.0


# ── Evaluator judge plumbing ─────────────────────────────────────────────────


def test_invoke_llm_text_paths() -> None:
    assert _invoke_llm_text(ScriptedLLM(" 7 "), "p") == "7"
    assert _invoke_llm_text(lambda p: f"echo {p}", "hi") == "echo hi"
    with pytest.raises(TypeError):
        _invoke_llm_text(object(), "p")

    class Failing:
        def invoke(self, _messages):
            raise ConnectionError("judge down")

    with pytest.raises(ConnectionError):
        _invoke_llm_text(Failing(), "p")


def test_evaluator_with_llm_judge() -> None:
    ev = RAGEvaluator(llm=ScriptedLLM("8", "6"))
    score = ev.evaluate("q?", "answer", ["answer context"], ground_truth="answer")
    assert score.used_llm_judge
    assert score.faithfulness_score == 0.8 and score.answer_relevance == 0.6


def test_faithfulness_score_logs_judge_failure(caplog) -> None:
    def judge(_a: str, _c: str) -> float:
        raise RuntimeError("boom")

    assert RAGEvaluator().faithfulness_score("a", "c", judge=judge) is None
    assert "judge failed" in caplog.text


# ── Retrieval helpers ────────────────────────────────────────────────────────


def test_invoke_retriever_and_shims() -> None:
    docs = [Document(page_content=str(i)) for i in range(5)]

    class Legacy:
        def get_relevant_documents(self, q: str) -> list[Document]:
            return docs[:2]

    assert len(invoke_retriever(ListRetriever(docs), "q")) == 5
    assert len(invoke_retriever(Legacy(), "q")) == 2
    with pytest.raises(TypeError, match="neither invoke"):
        invoke_retriever(object(), "q")

    shim = RetrieverAsVectorStore(ListRetriever(docs))  # type: ignore[arg-type]
    assert len(shim.similarity_search("q", k=3)) == 3
    assert shim.as_retriever() is shim._retriever

    guard = ReRankerGuard(rerank_fn=lambda q, d: list(reversed(d)), top_n=2)
    assert [d.page_content for d in guard.apply("q", docs)] == ["4", "3"]
    assert ReRankerGuard().apply("q", docs) == docs


def test_content_hash_is_stable() -> None:
    assert content_hash("abc") == "ba7816bf8f01cfea"


# ── Factories (optional deps faked via sys.modules) ──────────────────────────


def _fake_module(monkeypatch, name: str, **attrs) -> None:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)


class _Recorder:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


def test_llm_factories_pass_arguments(monkeypatch) -> None:
    _fake_module(monkeypatch, "langchain_openai", ChatOpenAI=_Recorder, OpenAIEmbeddings=_Recorder)
    _fake_module(monkeypatch, "langchain_anthropic", ChatAnthropic=_Recorder)
    _fake_module(monkeypatch, "langchain_mistralai", ChatMistralAI=_Recorder)

    assert llm_factory.create_openai_llm("gpt-4o-mini").kwargs["model"] == "gpt-4o-mini"
    assert llm_factory.create_anthropic_llm(api_key="k").kwargs["api_key"] == "k"
    m = llm_factory.create_mistral_llm(api_key="k", load_dotenv=False)
    assert m.kwargs["model"] == "mistral-small-latest"
    emb = embedding_factory.create_openai_embeddings(model="text-embedding-3-small")
    assert emb.kwargs["model"] == "text-embedding-3-small"


def test_llm_factories_missing_dependency_messages(monkeypatch) -> None:
    for name in ("langchain_openai", "langchain_anthropic", "langchain_mistralai"):
        monkeypatch.setitem(sys.modules, name, None)  # makes `import name` raise ImportError
    with pytest.raises(ImportError, match="pip install langchain-openai"):
        llm_factory.create_openai_llm()
    with pytest.raises(ImportError, match="pip install langchain-anthropic"):
        llm_factory.create_anthropic_llm()
    with pytest.raises(ImportError, match="langchain-mistralai"):
        llm_factory.create_mistral_llm(api_key="k", load_dotenv=False)
    with pytest.raises(ImportError, match="langchain-openai"):
        embedding_factory.create_openai_embeddings()


def test_mistral_and_env_require_key(monkeypatch) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_API_TOKEN", raising=False)
    with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
        llm_factory.create_mistral_llm(load_dotenv=False)
    with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
        llm_factory.create_llm_from_env(load_dotenv=False)

    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    monkeypatch.setenv("MISTRAL_MODEL", "mistral-large-latest")
    assert env.mistral_config_from_env() == ("k", "mistral-large-latest")


def test_load_env_reads_dotenv_file(tmp_path, monkeypatch) -> None:
    pytest.importorskip("dotenv")
    f = tmp_path / ".env"
    f.write_text("RAGFALLBACK_TEST_VAR=hello\n")
    monkeypatch.delenv("RAGFALLBACK_TEST_VAR", raising=False)
    assert env.load_env(str(f)) is True
    import os

    assert os.environ["RAGFALLBACK_TEST_VAR"] == "hello"
    monkeypatch.delenv("RAGFALLBACK_TEST_VAR")


def test_anthropic_embeddings_not_supported() -> None:
    with pytest.raises(NotImplementedError):
        embedding_factory.create_anthropic_embeddings()


def test_faiss_factory_persists_and_reloads(tmp_path, caplog) -> None:
    pytest.importorskip("faiss")
    docs = [Document(page_content="refund policy"), Document(page_content="office hours")]
    target = tmp_path / "idx"
    vs = vector_store_factory.create_faiss_vector_store(docs, KeywordEmbeddings(), str(target))
    assert (target / "index.faiss").exists()
    assert vs.similarity_search("refund", k=1)[0].page_content == "refund policy"

    # Existing index + no opt-in: rebuild, but say so instead of failing silently.
    vector_store_factory.create_faiss_vector_store(docs[:1], KeywordEmbeddings(), str(target))
    assert "allow_dangerous_deserialization=False" in caplog.text

    reloaded = vector_store_factory.create_faiss_vector_store(
        [], KeywordEmbeddings(), str(target), allow_dangerous_deserialization=True
    )
    assert len(reloaded.similarity_search("refund", k=5)) == 1


def test_vector_store_factories_missing_dependency(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "langchain_community.vectorstores", None)
    with pytest.raises(ImportError, match="faiss-cpu"):
        vector_store_factory.create_faiss_vector_store([], KeywordEmbeddings())
    with pytest.raises(ImportError, match="chromadb"):
        vector_store_factory.create_chroma_vector_store([], KeywordEmbeddings())
    with pytest.raises(ImportError, match="qdrant-client"):
        vector_store_factory.create_qdrant_vector_store([], KeywordEmbeddings())


def test_ollama_factories(monkeypatch) -> None:
    _fake_module(monkeypatch, "langchain_community.llms", Ollama=_Recorder)
    _fake_module(monkeypatch, "langchain_community.embeddings", OllamaEmbeddings=_Recorder)
    llm = llm_factory.create_open_source_llm(model="llama3")
    assert llm.kwargs == {"model": "llama3", "base_url": "http://localhost:11434", "temperature": 0}
    emb = embedding_factory.create_ollama_embeddings(base_url="http://h:1")
    assert emb.kwargs["base_url"] == "http://h:1"
