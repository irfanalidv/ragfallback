"""Offline tests for the diagnostics subpackage (fake embeddings, temp files)."""

from __future__ import annotations

import hashlib
import math

import pytest
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from ragfallback.diagnostics import (
    CONTEXT_WINDOW_TOKEN_PRESETS,
    ChunkQualityChecker,
    ContextWindowGuard,
    EmbeddingGuard,
    EmbeddingQualityProbe,
    EmbeddingValidator,
    OverlappingContextStitcher,
    RetrievalHealthCheck,
    StaleIndexDetector,
    infer_vectorstore_embedding_dim,
    sanitize_documents,
    sanitize_metadata,
)
from ragfallback.diagnostics.context_window import _estimate_tokens
from ragfallback.diagnostics.retrieval_health import _doc_key
from tests.unit.fakes import KeywordEmbeddings, StoreOf

CORPUS = [
    Document(page_content="Refunds are issued within 14 days of purchase.", metadata={"id": "r"}),
    Document(
        page_content="API tokens expire after 90 days and must be rotated.", metadata={"id": "t"}
    ),
    Document(page_content="Our office is closed on public holidays.", metadata={"id": "h"}),
]


@pytest.fixture
def store() -> InMemoryVectorStore:
    vs = InMemoryVectorStore(embedding=KeywordEmbeddings())
    vs.add_documents(CORPUS)
    return vs


# ── ContextWindowGuard ────────────────────────────────────────────────────────


def test_context_window_ranks_by_similarity_and_trims_to_budget() -> None:
    docs = [
        Document(page_content="holiday schedule " * 40),
        Document(page_content="refund policy refund days " * 10),
        Document(page_content="refund " * 5),
    ]
    guard = ContextWindowGuard(max_context_tokens=80)
    selected, report = guard.select("refund policy", docs, KeywordEmbeddings())
    assert selected[0].page_content.startswith("refund policy")
    assert report.used_tokens <= 80
    assert report.n_selected + report.n_dropped == 3
    assert report.n_dropped >= 1 and report.dropped_reasons
    assert "chunks" in report.summary()


def test_context_window_empty_and_lost_in_middle() -> None:
    guard = ContextWindowGuard(max_context_tokens=10_000)
    selected, report = guard.select("q", [], KeywordEmbeddings())
    assert selected == [] and report.dropped_reasons == ["no documents"]

    docs = [Document(page_content=f"refund {'x ' * i}") for i in range(4)]
    plain, _ = guard.select("refund", docs, KeywordEmbeddings())
    reordered, _ = guard.select("refund", docs, KeywordEmbeddings(), lost_in_middle=True)
    assert reordered[0] is plain[0]
    assert reordered[-1] is plain[1]


def test_context_window_presets() -> None:
    exact = ContextWindowGuard.from_model_name("gpt-4")
    assert exact.max_context_tokens == int(CONTEXT_WINDOW_TOKEN_PRESETS["gpt-4"] * 0.88)
    prefixed = ContextWindowGuard.from_model_name("mistral-large-latest", reserve_ratio=0)
    assert prefixed.max_context_tokens == CONTEXT_WINDOW_TOKEN_PRESETS["mistral-large"]
    generic = ContextWindowGuard.from_model_name("open-mistral-7b", reserve_ratio=0)
    assert generic.max_context_tokens == CONTEXT_WINDOW_TOKEN_PRESETS["default"]
    unknown = ContextWindowGuard.from_model_name("some-new-model", reserve_ratio=0)
    assert unknown.max_context_tokens == CONTEXT_WINDOW_TOKEN_PRESETS["default"]


def test_estimate_tokens_char_heuristic() -> None:
    assert _estimate_tokens("") == 0
    assert _estimate_tokens("a" * 40) == 10
    # Unknown tiktoken model falls back to the char heuristic instead of raising.
    assert _estimate_tokens("a" * 40, encoding_name="not-a-model") == 10


# ── Embedding probe / validator / guard ──────────────────────────────────────


def test_embedding_probe_ok_and_warnings() -> None:
    probe = EmbeddingQualityProbe(min_mean_top1=0.5, max_spread=1.0)
    good = probe.run(KeywordEmbeddings(), "refund policy", ["refund policy details"])
    assert good.ok and good.mean_top1_similarity > 0.5
    bad = probe.run(KeywordEmbeddings(), "refund policy", ["zebra giraffe"])
    assert not bad.ok and "best-match" in bad.warnings[0]
    assert "embedding_probe" in bad.summary()
    assert not probe.run(KeywordEmbeddings(), "q", []).ok
    assert probe.run(KeywordEmbeddings(), "q", ["  "]).warnings == ["all reference snippets empty"]


def test_embedding_probe_spread_warning() -> None:
    probe = EmbeddingQualityProbe(min_mean_top1=0.0, max_spread=0.1)
    rep = probe.run(KeywordEmbeddings(), "refund policy", ["refund policy", "zebra giraffe"])
    assert any("spread" in w for w in rep.warnings)


class _FakeCollection:
    def __init__(self, meta: dict) -> None:
        self._meta = meta

    def metadata(self) -> dict:
        return self._meta


class _FakeChroma:
    def __init__(self, dim: int) -> None:
        self._collection = _FakeCollection({"dimension": dim})


class _FakeFaissIndex:
    d = 64


class _FakeFaiss:
    index = _FakeFaissIndex()


def test_infer_vectorstore_dim_and_validator() -> None:
    assert infer_vectorstore_embedding_dim(_FakeChroma(384)) == 384
    assert infer_vectorstore_embedding_dim(_FakeFaiss()) == 64
    assert infer_vectorstore_embedding_dim(object()) is None

    validator = EmbeddingValidator()
    assert validator.validate(KeywordEmbeddings(64), _FakeFaiss()).ok
    mismatch = validator.validate(KeywordEmbeddings(64), _FakeChroma(384))
    assert not mismatch.ok and "Model / index mismatch" in mismatch.message
    unknown = validator.validate(KeywordEmbeddings(64), object())
    assert unknown.ok and unknown.store_reported_dim is None


def test_embedding_guard_batches_and_raw_vectors() -> None:
    guard = EmbeddingGuard(expected_dim=64)
    assert guard.validate_document_batch(KeywordEmbeddings(64), ["a", "b"]).ok
    assert not guard.validate_document_batch(KeywordEmbeddings(32), ["a"]).ok
    assert not guard.validate_document_batch(KeywordEmbeddings(64), []).ok

    assert EmbeddingGuard.validate_raw_vectors([[1.0, 0.0], [0.0, 1.0]], expected_dim=2).ok
    assert not EmbeddingGuard.validate_raw_vectors([[1.0], [1.0, 2.0]]).ok
    assert not EmbeddingGuard.validate_raw_vectors([[0.0, 0.0]]).ok
    assert not EmbeddingGuard.validate_raw_vectors([[math.nan, 1.0]]).ok
    assert not EmbeddingGuard.validate_raw_vectors([[1.0, 2.0]], expected_dim=3).ok
    assert not EmbeddingGuard.validate_raw_vectors([]).ok


# ── RetrievalHealthCheck ──────────────────────────────────────────────────────


def test_retrieval_health_gold_keys(store: InMemoryVectorStore) -> None:
    rep = RetrievalHealthCheck(k=1).run(
        store,
        [
            {"query": "refunds purchase", "gold_keys": {"id:r"}},
            {"query": "api tokens expire", "gold_keys": ["id:t"]},
            {"query": "missing gold"},
        ],
    )
    assert rep.ok and rep.cases_run == 2 and rep.mean_recall_at_k == 1.0
    assert any("skipped" in n for n in rep.notes)
    assert "mean_recall@1" in rep.summary()


def test_retrieval_health_substring_probes(store: InMemoryVectorStore) -> None:
    rep = RetrievalHealthCheck(k=1).run_substring_probes(
        store,
        {"refunds purchase": "14 DAYS", "office holidays": "tokens"},
    )
    assert rep.hit_rate == 0.5 and not rep.ok
    assert rep.avg_latency_ms is not None


def test_retrieval_health_records_retriever_errors() -> None:
    class Boom:
        def invoke(self, q: str) -> list:
            raise RuntimeError("index offline")

    rep = RetrievalHealthCheck().run_substring_probes(StoreOf(Boom()), {"q": "x"})
    assert rep.hit_rate == 0.0 and "index offline" in rep.notes[0]


def test_retrieval_health_quick_check(store: InMemoryVectorStore) -> None:
    docs = [Document(page_content=d.page_content * 2) for d in CORPUS]
    vs = InMemoryVectorStore(embedding=KeywordEmbeddings())
    vs.add_documents(docs)
    rep = RetrievalHealthCheck(k=2).quick_check(vs, docs, seed=0)
    assert rep.cases_run > 0 and rep.hit_rate > 0
    assert not RetrievalHealthCheck().quick_check(vs, []).ok


def test_doc_key_is_process_stable() -> None:
    doc = Document(page_content="no metadata here")
    # sha256-based, so the value is fixed across interpreter runs (unlike hash()).
    expected = hashlib.sha256(b"no metadata here").hexdigest()[:16]
    assert _doc_key(doc) == f"hash:{expected}"
    assert _doc_key(Document(page_content="x", metadata={"source": "a.md"})) == "source:a.md"


# ── StaleIndexDetector ────────────────────────────────────────────────────────


def test_stale_index_detects_changed_removed_and_new_files(tmp_path) -> None:
    a, b, c = tmp_path / "a.md", tmp_path / "b.md", tmp_path / "c.md"
    a.write_text("alpha")
    b.write_text("beta")
    det = StaleIndexDetector(manifest_path=tmp_path / "manifest.json")

    missing = det.check_paths([a, b])
    assert not missing.ok and "No saved manifest" in missing.summary()

    det.record_paths([a, b])
    assert det.check_paths([a, b]).ok

    a.write_text("alpha v2")
    b.unlink()
    c.write_text("gamma")
    rep = det.check_paths([a, b, c])
    assert rep.has_stale
    assert len(rep.stale_paths) == 1 and len(rep.missing_paths) == 1 and len(rep.new_paths) == 1
    assert "1 changed" in rep.message and "1 removed" in rep.message and "1 new" in rep.message


def test_stale_index_document_contents(tmp_path) -> None:
    det = StaleIndexDetector(manifest_path=tmp_path / "m.json")
    docs = [Document(page_content="v1", metadata={"source": "faq"})]
    assert not det.check_document_contents(docs).ok
    det.record_document_contents(docs)
    assert det.check_document_contents(docs).ok
    changed = det.check_document_contents([Document(page_content="v2", metadata={"source": "faq"})])
    assert changed.stale_paths == ["faq::0"]


def test_stale_index_record_from_documents(tmp_path) -> None:
    f = tmp_path / "doc.txt"
    f.write_text("hello")
    det = StaleIndexDetector(manifest_path=tmp_path / "m.json")
    det.record_from_documents([Document(page_content="x", metadata={"source": str(f)})])
    assert det.check_paths([f]).ok


# ── Stitcher / sanitizer / chunk checker ─────────────────────────────────────


def _chunk(src: str, idx: int | None, text: str) -> Document:
    md = {"source": src}
    if idx is not None:
        md["chunk_index"] = idx
    return Document(page_content=text, metadata=md)


def test_stitcher_merges_consecutive_chunks_per_source() -> None:
    docs = [
        _chunk("hr.md", 2, "C"),
        _chunk("hr.md", 0, "A"),
        _chunk("hr.md", 1, "B"),
        _chunk("hr.md", 5, "F"),
        _chunk("faq.md", None, "loose"),
    ]
    out = OverlappingContextStitcher(separator=" ").stitch(docs)
    texts = sorted(d.page_content for d in out)
    assert texts == ["A B C", "F", "loose"]
    merged = next(d for d in out if d.page_content == "A B C")
    assert merged.metadata["stitched_chunk_start"] == 0
    assert merged.metadata["stitched_chunk_end"] == 2
    assert merged.metadata["ragfallback_stitched"] is True
    assert OverlappingContextStitcher().stitch([]) == []


def test_stitcher_expand_neighbor_indices() -> None:
    st = OverlappingContextStitcher()
    docs = [_chunk("a", 3, "d"), _chunk("a", 1, "b")]
    assert [d.page_content for d in st.expand_neighbor_indices(docs)] == ["b", "d"]
    assert st.expand_neighbor_indices(docs, neighbor_radius=0) == docs
    no_idx = [_chunk("a", None, "x")]
    assert st.expand_neighbor_indices(no_idx) == no_idx


def test_sanitize_metadata_and_documents() -> None:
    md = sanitize_metadata(
        {"tags": ["a", "b"], "n": 3, "nested": {"k": 1}, "none": None, "b": b"x"}
    )
    assert isinstance(md["tags"], str) and md["n"] == 3
    assert all(isinstance(v, (str, int, float, bool)) or v is None for v in md.values())
    docs = sanitize_documents([Document(page_content="x", metadata={"tags": ["a"]})])
    assert isinstance(docs[0].metadata["tags"], str)


def test_chunk_checker_flags_and_auto_fix() -> None:
    checker = ChunkQualityChecker(min_chars=20, max_chars=60, require_sentence_end=True)
    chunks = ["tiny", "x" * 100 + ".", "A proper sentence that is long enough to pass."]
    rep = checker.check(chunks)
    assert rep.has_issues
    assert any("too short" in v for v in rep.violations)
    assert any("too long" in v for v in rep.violations)
    assert checker.suggest_fixes(rep)
    fixed = checker.auto_fix(chunks, target_chars=40, overlap_chars=5)
    assert all(len(t) <= 60 for t in fixed)
    assert not ChunkQualityChecker().check([]).ok


def test_auto_fix_folds_trailing_fragment_and_prefers_sentence_cuts() -> None:
    from ragfallback.diagnostics.chunking import _merge_small_chunks, _split_large_text

    assert _merge_small_chunks(["a" * 100, "tiny"], min_chars=50) == ["a" * 100 + " tiny"]
    text = "First sentence is here. Second sentence follows it. Third one ends the text."
    parts = _split_large_text(text, target=40, overlap=0)
    assert parts[0] == "First sentence is here."
    assert "".join(parts).replace(" ", "") == text.replace(" ", "")


@pytest.mark.parametrize(
    "model, preset_key",
    [
        ("gpt-4-turbo-2024-04-09", "gpt-4-turbo"),  # longest prefix, not "gpt-4"
        ("gpt-4o-mini-2024-07-18", "gpt-4o-mini"),
        ("gpt-4.1-mini", "gpt-4.1"),
        ("gpt-5-mini", "gpt-5"),
        ("gpt-5.4-mini", "gpt-5.4-mini"),
        ("gpt-5.6-terra", "gpt-5"),  # not listed: falls back to the smaller gpt-5 budget
        ("o3-mini", "o3"),
        ("claude-opus-4-5", "claude-opus-4"),  # 200K
        ("claude-opus-4-6", "claude-opus-4-6"),  # 1M
        ("claude-sonnet-4-5-20250929", "claude-sonnet-4"),
        ("claude-sonnet-5-5", "claude-sonnet-5"),
        ("mistral-small-latest", "mistral-small"),
        ("ministral-8b-2512", "ministral"),
        ("codestral-latest", "codestral"),
    ],
)
def test_context_presets_resolve_by_longest_prefix(model: str, preset_key: str) -> None:
    guard = ContextWindowGuard.from_model_name(model, reserve_ratio=0)
    assert guard.max_context_tokens == CONTEXT_WINDOW_TOKEN_PRESETS[preset_key]


def test_context_presets_are_dated() -> None:
    from ragfallback.diagnostics.context_window import CONTEXT_PRESETS_AS_OF

    assert CONTEXT_PRESETS_AS_OF[4] == "-" and len(CONTEXT_PRESETS_AS_OF) == 10
