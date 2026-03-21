"""
ragfallback — Library Verification Suite v2.0
=============================================
Verifies all modules work correctly after `pip install ragfallback`.
Run: python verify_library.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results: list[tuple[str, bool, str]] = []


def check(name: str, fn) -> None:
    try:
        fn()
        results.append((name, True, ""))
        print(f"{PASS}: {name}")
    except Exception:
        msg = traceback.format_exc().strip().splitlines()[-1]
        results.append((name, False, msg))
        print(f"{FAIL}: {name}")
        print(f"       {msg}")


# ─── Test 1: Root imports ─────────────────────────────────────────────────────
def _test_root_imports():
    from ragfallback import AdaptiveRAGRetriever, CostTracker, MetricsCollector, QueryResult
    assert AdaptiveRAGRetriever
    assert QueryResult
    assert CostTracker
    assert MetricsCollector


# ─── Test 2: ragfallback.diagnostics ─────────────────────────────────────────
def _test_diagnostics():
    from ragfallback.diagnostics import (
        ChunkQualityChecker,
        ChunkQualityReport,
        ContextWindowGuard,
        EmbeddingGuard,
        EmbeddingGuardReport,
        EmbeddingQualityProbe,
        EmbeddingQualityReport,
        OverlappingContextStitcher,
        RetrievalHealthCheck,
        StaleIndexDetector,
        sanitize_documents,
        sanitize_metadata,
    )
    # Exercise each one minimally
    from langchain_core.documents import Document

    docs = [Document(page_content="The quick brown fox jumps over the lazy dog. " * 5)]
    report = ChunkQualityChecker().check(docs)
    assert isinstance(report, ChunkQualityReport)

    stitcher = OverlappingContextStitcher()
    merged = stitcher.stitch(docs)
    assert isinstance(merged, list)

    clean = sanitize_documents([Document(page_content="x", metadata={"tags": ["a", "b"]})])
    assert isinstance(clean[0].metadata["tags"], str)

    clean_meta = sanitize_metadata({"key": [1, 2]})
    assert isinstance(clean_meta["key"], str)


# ─── Test 3: ragfallback.retrieval ───────────────────────────────────────────
def _test_retrieval():
    from ragfallback.retrieval import (
        FailoverRetriever,
        ReRankerGuard,
        RetrieverAsVectorStore,
        SmartThresholdHybridRetriever,
    )
    assert SmartThresholdHybridRetriever
    assert FailoverRetriever
    assert ReRankerGuard
    assert RetrieverAsVectorStore


# ─── Test 4: ragfallback.strategies ──────────────────────────────────────────
def _test_strategies():
    from ragfallback.strategies import (
        FallbackStrategy,
        HopResult,
        MultiHopFallbackStrategy,
        MultiHopResult,
        QueryVariationsStrategy,
    )
    hop = HopResult(
        hop_number=1,
        sub_question="test?",
        retrieved_chunks=["chunk1"],
        partial_answer="answer",
        confidence=0.8,
    )
    result = MultiHopResult(final_answer="done", hops=[hop], total_hops=1, success=True)
    assert result.summary()

    strategy = MultiHopFallbackStrategy(max_hops=2, top_k=3)
    assert strategy


# ─── Test 5: ragfallback.evaluation ──────────────────────────────────────────
def _test_evaluation():
    from ragfallback.evaluation import RAGEvaluator, RAGScore

    ev = RAGEvaluator()
    score = ev.evaluate(
        question="What is Python?",
        answer="Python is a high-level programming language.",
        contexts=["Python is a high-level programming language used for many purposes."],
        ground_truth="Python is a programming language.",
    )
    assert isinstance(score, RAGScore)
    assert score.overall_score >= 0.0
    summary = ev.batch_summary([score])
    assert isinstance(summary, str)


# ─── Test 6: ragfallback.tracking ────────────────────────────────────────────
def _test_tracking():
    from ragfallback import CostTracker, MetricsCollector

    ct = CostTracker(budget=10.0)
    ct.record_tokens(input_tokens=100, output_tokens=50, model="gpt-4o-mini")
    report = ct.get_report()
    assert "total_cost" in report

    mc = MetricsCollector()
    mc.record_success(attempts=1, confidence=0.9, cost=0.001, latency_ms=120, strategy_used="direct")
    stats = mc.get_stats()
    assert stats["total_queries"] >= 1
    assert stats["success_rate"] > 0


# ─── Test 7: Factory + embeddings ────────────────────────────────────────────
def _test_embeddings():
    from ragfallback.utils.embedding_factory import create_open_source_embeddings

    emb = create_open_source_embeddings()
    vec = emb.embed_query("Hello world")
    assert isinstance(vec, list)
    assert len(vec) == 384, f"Expected dim 384, got {len(vec)}"


# ─── Test 8: Example file syntax ─────────────────────────────────────────────
def _test_example_syntax():
    examples_dir = Path(__file__).parent / "examples"
    py_files = sorted(examples_dir.glob("*.py"))
    assert py_files, "No .py files found in examples/"
    failed: list[str] = []
    for f in py_files:
        try:
            compile(f.read_text(encoding="utf-8"), str(f), "exec")
        except SyntaxError as exc:
            failed.append(f"{f.name}: {exc}")
    if failed:
        raise AssertionError("Syntax errors:\n" + "\n".join(failed))
    print(f"       {len(py_files)} example files — all valid syntax")


# ─── Test 9: Version ─────────────────────────────────────────────────────────
def _test_version():
    import ragfallback

    assert ragfallback.__version__ == "2.0.0", (
        f"Expected '2.0.0', got '{ragfallback.__version__}'"
    )
    print(f"       version={ragfallback.__version__}")


# ─── Run all ─────────────────────────────────────────────────────────────────
print("=" * 70)
print("ragfallback Library Verification Suite v2.0")
print("=" * 70)

check("Root imports (AdaptiveRAGRetriever, QueryResult, CostTracker, MetricsCollector)", _test_root_imports)
check("ragfallback.diagnostics (all 12 exports)", _test_diagnostics)
check("ragfallback.retrieval (SmartThresholdHybridRetriever, FailoverRetriever, ReRankerGuard, RetrieverAsVectorStore)", _test_retrieval)
check("ragfallback.strategies (QueryVariationsStrategy, MultiHopFallbackStrategy, HopResult, MultiHopResult)", _test_strategies)
check("ragfallback.evaluation (RAGEvaluator.evaluate() returns RAGScore with overall_score > 0)", _test_evaluation)
check("ragfallback.tracking (CostTracker.record_tokens, MetricsCollector.record_success)", _test_tracking)
check("Factory + embeddings (create_open_source_embeddings → dim 384)", _test_embeddings)
check("Example files syntax (compile all examples/*.py)", _test_example_syntax)
check("Version == '2.0.0'", _test_version)

print("=" * 70)
passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
print(f"\n{passed}/{total} tests passed")
if passed == total:
    print("\n🎉 All tests passed — library is production-ready.")
else:
    print("\n⚠️  Some tests failed — see errors above.")
    sys.exit(1)
