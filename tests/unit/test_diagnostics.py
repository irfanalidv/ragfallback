"""Tests for diagnostics and evaluation helpers."""

import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from ragfallback.diagnostics import ChunkQualityChecker, EmbeddingGuard, StaleIndexDetector
from ragfallback.diagnostics.retrieval_health import RetrievalHealthCheck
from ragfallback.evaluation import RAGEvaluator, recall_at_k
from ragfallback.exceptions import EmbeddingDimensionError


def test_chunk_quality_overlap():
    a = "hello world " * 20
    b = "world " * 5 + "extra content " * 30
    rep = ChunkQualityChecker(min_chars=10, max_chars=100000, target_overlap_ratio=0.01).check(
        [a, b]
    )
    assert rep.n_chunks == 2
    assert rep.estimated_overlap_ratio is not None
    assert rep.estimated_overlap_ratio >= 0


def test_chunk_too_short():
    rep = ChunkQualityChecker(min_chars=500).check(["hi"])
    assert not rep.ok
    assert rep.has_issues
    assert any("too short" in v for v in rep.violations)
    tips = ChunkQualityChecker().suggest_fixes(rep)
    assert tips


def test_embedding_guard_dimension():
    class FakeEmb:
        def embed_query(self, text: str):
            return [0.1, 0.2, 0.3]

        def embed_documents(self, texts):
            return [[0.6, 0.8, 0.0] for _ in texts]

    g = EmbeddingGuard(expected_dim=3)
    r = g.validate(FakeEmb())
    assert r.ok
    r2 = g.validate(FakeEmb(), expected_dim=128)
    assert not r2.ok
    assert "128" in r2.message and "mismatch" in r2.message.lower()
    try:
        r2.raise_if_failed()
        raise AssertionError("expected EmbeddingDimensionError")
    except EmbeddingDimensionError:
        pass


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "z"}, k=2) == 0.5
    assert recall_at_k([], {"a"}, k=5) == 0.0


def test_rag_evaluate_no_llm():
    ev = RAGEvaluator(llm=None)
    s = ev.evaluate(
        "What is Python?",
        "Python is a high-level language.",
        [Document(page_content="Python is a high-level programming language.")],
        ground_truth="Python is high-level",
    )
    assert 0.0 <= s.overall_score <= 1.0
    assert "Context precision" in s.report()


def test_embedding_guard_raw_vectors():
    rows = [[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]]
    r = EmbeddingGuard.validate_raw_vectors(rows, expected_dim=3)
    assert r.ok
    bad = EmbeddingGuard.validate_raw_vectors([[0.0, 0.0, 0.0]], expected_dim=3)
    assert not bad.ok


def test_rag_evaluator_heuristic():
    ev = RAGEvaluator()
    rep = ev.evaluate_heuristic(
        "Python is a high-level language.",
        [Document(page_content="Python is a high-level programming language.")],
    )
    assert rep.overall > 0.2
    assert "%" in rep.summary()


class _MockVSSubstring:
    def as_retriever(self, **kwargs):
        class R:
            def invoke(self, q):
                return [Document(page_content="Python is a high-level programming language.")]

        return R()


def test_retrieval_substring_probes():
    h = RetrievalHealthCheck(k=3)
    rep = h.run_substring_probes(
        _MockVSSubstring(),
        {"What is Python?": "high-level programming"},
        min_mean_hit_rate=0.5,
    )
    assert rep.hit_rate == 1.0
    assert rep.ok


def test_rag_evaluator_ndcg():
    ev = RAGEvaluator()
    # perfect binary relevance at top
    scores = [1.0, 1.0, 0.0, 0.0]
    m = ev.evaluate_ranked_relevance(scores, k=4)
    assert m["ndcg@4"] == pytest.approx(1.0, rel=1e-6)


def test_stale_index_detector(tmp_path: Path):
    f1 = tmp_path / "a.txt"
    f1.write_text("v1", encoding="utf-8")
    det = StaleIndexDetector()
    m1 = det.build_manifest([f1])
    man_path = tmp_path / "m.json"
    det.save_manifest(m1, man_path)
    r_ok = det.diff([f1], saved=json.loads(man_path.read_text(encoding="utf-8")))
    assert r_ok.ok
    f1.write_text("v2", encoding="utf-8")
    r_stale = det.diff([f1], saved=json.loads(man_path.read_text(encoding="utf-8")))
    assert not r_stale.ok
    assert r_stale.stale_paths
