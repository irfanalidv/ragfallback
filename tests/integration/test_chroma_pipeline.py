"""Integration tests: real Chroma + sentence-transformers (optional deps)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES = REPO_ROOT / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))


@pytest.mark.integration
def test_chroma_pipeline_retrieval_health(tmp_path):
    pytest.importorskip("chromadb")
    pytest.importorskip("sentence_transformers")
    import _kb_common

    from ragfallback.diagnostics import RetrievalHealthCheck

    docs = _kb_common.load_sample_kb(kb_dir=REPO_ROOT / "examples" / "sample_kb")
    assert docs, "sample_kb should yield chunks"
    persist = tmp_path / "chroma"
    vs = _kb_common.build_chroma_store(
        docs, persist_directory=persist, collection_name="itest_health"
    )
    health = RetrievalHealthCheck(k=4)
    report = health.quick_check(vs, docs, sample_size=min(10, len(docs)), seed=42)
    assert report.hit_rate >= 0.60
    if report.avg_latency_ms is not None:
        assert report.avg_latency_ms < 5000.0


@pytest.mark.integration
def test_embedding_guard_with_real_embeddings():
    pytest.importorskip("sentence_transformers")
    import _kb_common

    from ragfallback.diagnostics import EmbeddingGuard

    emb = _kb_common.get_embeddings()
    ok_rep = EmbeddingGuard(expected_dim=384).validate(emb)
    assert ok_rep.ok

    bad = EmbeddingGuard.validate_raw_vectors([[0.1] * 128, [0.2] * 128], expected_dim=384)
    assert not bad.ok


@pytest.mark.integration
def test_chunk_quality_on_real_kb():
    import _kb_common

    from ragfallback.diagnostics import ChunkQualityChecker

    texts = [d.page_content for d in _kb_common.load_sample_kb(kb_dir=REPO_ROOT / "examples" / "sample_kb")]
    rep = ChunkQualityChecker(
        min_chars=30,
        min_words=4,
        target_overlap_ratio=0.0,
        overlap_tolerance=1.0,
        mid_sentence_warn_ratio=2.0,
    ).check(texts)
    assert rep.ok, rep.summary()


@pytest.mark.integration
def test_context_window_never_overflows(tmp_path):
    pytest.importorskip("chromadb")
    pytest.importorskip("sentence_transformers")
    import _kb_common

    from ragfallback.diagnostics import ContextWindowGuard

    docs = _kb_common.load_sample_kb(kb_dir=REPO_ROOT / "examples" / "sample_kb")
    persist = tmp_path / "chroma_ctx"
    vs = _kb_common.build_chroma_store(docs, persist_directory=persist, collection_name="itest_ctx")
    emb = _kb_common.get_embeddings()
    guard = ContextWindowGuard.from_model_name("gpt-3.5-turbo")
    queries = ["refund policy", "API authentication", "rate limits"]
    for q in queries:
        retrieved = vs.similarity_search(q, k=min(20, max(5, len(docs))))
        _packed, rep = guard.select(q, retrieved, emb)
        assert rep.used_tokens <= rep.budget_tokens, rep.summary()
