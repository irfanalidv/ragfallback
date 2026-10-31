"""Offline tests for ragfallback.mlops (no ragas / mlflow / network required)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from ragfallback import AdaptiveRAGRetriever
from ragfallback.mlops import (
    BaselineRegistry,
    GoldenReport,
    GoldenRunner,
    MLflowLogger,
    QuerySimulator,
    RagasHook,
    RagasReport,
    RegressionError,
    generate_locustfile,
)
from ragfallback.strategies import QueryVariationsStrategy
from tests.unit.fakes import KeywordEmbeddings, ScriptedLLM


def _ragas(score: float = 0.8) -> RagasReport:
    return RagasReport(
        faithfulness=score,
        answer_relevance=score,
        context_precision=score,
        context_recall=score,
        n_samples=1,
        fallback_mode=True,
        timestamp=datetime.now(timezone.utc),
        raw={},
    )


def _report(score: float = 0.8, recall: float = 1.0, p95: float = 100.0) -> GoldenReport:
    return GoldenReport(
        ragas=_ragas(score),
        recall_at_3=recall,
        recall_at_5=recall,
        latency_p95_ms=p95,
        latency_mean_ms=p95 / 2,
        fallback_rate=0.0,
        n_samples=1,
        timestamp=datetime.now(timezone.utc),
        per_sample=[],
    )


# ── BaselineRegistry ─────────────────────────────────────────────────────────


def test_baseline_registry_roundtrip_and_gate(tmp_path) -> None:
    path = tmp_path / "baselines.json"
    reg = BaselineRegistry(str(path))
    assert not reg.exists("squad")
    reg.compare_or_fail(_report(), dataset="squad")  # no baseline -> warn only

    reg.update(_report(0.8), dataset="squad")
    saved = json.loads(path.read_text())
    assert saved["squad"]["faithfulness"] == 0.8
    assert saved["squad"]["recorded_at"].endswith("Z")

    reloaded = BaselineRegistry(str(path))
    reloaded.compare_or_fail(_report(0.79), dataset="squad")  # within 5%
    with pytest.raises(RegressionError, match="faithfulness"):
        reloaded.compare_or_fail(_report(0.5), dataset="squad")
    with pytest.raises(RegressionError, match="latency_p95_ms"):
        reloaded.compare_or_fail(_report(p95=200.0), dataset="squad")
    reloaded.compare_or_fail(_report(p95=110.0), dataset="squad", latency_threshold=0.2)
    with pytest.raises(KeyError):
        reloaded.get("other")


def test_baseline_registry_rejects_bad_input(tmp_path) -> None:
    path = tmp_path / "b.json"
    path.write_text("{not json")
    reg = BaselineRegistry(str(path))
    assert not reg.exists("x")
    with pytest.raises(TypeError):
        reg.update({"faithfulness": 1.0}, dataset="x")  # type: ignore[arg-type]


# ── QuerySimulator ───────────────────────────────────────────────────────────


def test_query_simulator_transforms() -> None:
    sim = QuerySimulator(ood_topics=["What is the weather?"])
    rows = sim.simulate_unhappy_paths(["How do Acme API tokens expire after rotation?"])
    by_type = {r.query_type: r.transformed for r in rows}
    assert set(by_type) == {"short_keyword", "long_nl", "ambiguous", "out_of_domain"}
    assert by_type["short_keyword"] == "Acme API"
    assert by_type["long_nl"].startswith("Please explain in detail:")
    assert "Acme" not in by_type["ambiguous"]
    assert by_type["out_of_domain"] == "What is the weather?"


def test_query_simulator_mix_validation() -> None:
    sim = QuerySimulator()
    out = sim.simulate(["a b c"], mix={"long_nl": 1.0})
    assert out[0].query_type == "long_nl"
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        sim.simulate(["q"], mix={"long_nl": 0.5})


# ── RagasHook (heuristic fallback) ───────────────────────────────────────────


def test_ragas_hook_fallback_scores() -> None:
    hook = RagasHook(llm=None, embeddings=None)
    hook._ragas_available = False  # force the offline path even if ragas is installed
    empty = hook.evaluate_sync([])
    assert empty.n_samples == 0
    rep = hook.evaluate_sync(
        [
            {
                "question": "When do refunds arrive?",
                "answer": "Refunds are issued within 14 days.",
                "contexts": ["Refunds are issued within 14 days of purchase."],
                "ground_truth": "14 days",
            }
        ]
    )
    assert rep.fallback_mode and rep.n_samples == 1
    assert rep.context_recall == 1.0
    assert 0.0 < rep.faithfulness <= 1.0


# ── GoldenRunner end to end (fake LLM + in-memory store) ─────────────────────


def _retriever() -> AdaptiveRAGRetriever:
    vs = InMemoryVectorStore(embedding=KeywordEmbeddings())
    vs.add_documents(
        [
            Document(page_content="Refunds are issued within 14 days.", metadata={"id": "refund"}),
            Document(page_content="Office closed on public holidays.", metadata={"id": "office"}),
        ]
    )
    llm = ScriptedLLM('{"answer": "Within 14 days", "source": "refund"}')
    return AdaptiveRAGRetriever(
        vector_store=vs,
        llm=llm,  # type: ignore[arg-type]
        embedding_model=KeywordEmbeddings(),
        fallback_strategies=[QueryVariationsStrategy(num_variations=0)],
        max_attempts=2,
        min_confidence=0.0,
        enable_logging=False,
    )


GOLDEN = [
    {"query": "When are refunds issued?", "ground_truth": "14 days", "relevant_doc_ids": ["refund"]}
]


def test_golden_runner_sync(tmp_path) -> None:
    hook = RagasHook(llm=None, embeddings=None)
    hook._ragas_available = False
    data = tmp_path / "golden.json"
    data.write_text(json.dumps(GOLDEN))
    report = GoldenRunner(_retriever(), hook, str(data)).run()
    assert report.n_samples == 1
    assert report.recall_at_3 == 1.0 and report.recall_at_5 == 1.0
    assert report.per_sample[0]["answer"] == "Within 14 days"
    assert report.latency_p95_ms >= 0 and report.cache_stats is None


async def test_golden_runner_async_with_cache_monitor() -> None:
    from ragfallback.tracking import CacheMonitor

    hook = RagasHook(llm=None, embeddings=None)
    hook._ragas_available = False
    monitor = CacheMonitor(max_size=8)
    report = await GoldenRunner(_retriever(), hook, GOLDEN * 2, cache_monitor=monitor).run_async()
    assert report.n_samples == 2 and report.recall_at_3 == 1.0
    assert report.cache_stats is not None and report.cache_stats["cache_misses"] >= 1


def test_golden_runner_rejects_empty_dataset() -> None:
    with pytest.raises(ValueError, match="empty"):
        GoldenRunner(_retriever(), RagasHook(llm=None, embeddings=None), [])


# ── Locust + MLflow ──────────────────────────────────────────────────────────


def test_generate_locustfile(tmp_path) -> None:
    out = tmp_path / "locustfile.py"
    generate_locustfile(str(out), endpoint="http://localhost:8000")
    text = out.read_text()
    compile(text, str(out), "exec")  # generated file is valid Python
    assert "HttpUser" in text and "http://localhost:8000" in text


def test_mlflow_logger_is_noop_without_mlflow(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name == "mlflow":
            raise ImportError("no mlflow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    logger = MLflowLogger()
    logger.log_golden_report(_report(), run_name="t")  # must not raise
