"""
ci_regression_gate.py — ragfallback MLOps CI Gate
==================================================
This is what GitHub Actions runs on every push to main.

What it does:
  1. Loads golden_qa.json (built from SQuAD by build_golden_dataset.py)
  2. Builds a fresh ChromaDB index from the same passages
  3. Runs GoldenRunner (async) — measures recall@3, recall@5, latency P95
  4. Compares against baselines.json (committed to repo)
  5. Exits 0 (pass) or 1 (regression detected)

  compare_or_fail(..., threshold=0.05, latency_threshold=0.12) so quality
  metrics stay at 5% while P95 tolerates shared-runner noise.

No LLM API keys required. Heuristic eval mode.
All compute is local (all-MiniLM-L6-v2 embeddings, ChromaDB).

Run locally: python examples/ci_regression_gate.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if (_repo_root / "ragfallback").is_dir() and str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

_examples_dir = Path(__file__).resolve().parent

# JSON answer shape expected by AdaptiveRAGRetriever's response parser
_HEURISTIC_JSON = '{"answer": "See retrieved context.", "source": "retriever"}'


def _build_store_and_retriever(docs_registry_path: Path):
    """Build ChromaDB + AdaptiveRAGRetriever from passage registry."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from ragfallback import AdaptiveRAGRetriever, CostTracker, MetricsCollector

    # Load passage registry (squad only for CI speed)
    docs_raw = json.loads(docs_registry_path.read_text())
    squad_docs = [d for d in docs_raw if d["source"] == "squad"][:50]
    docs = [
        Document(
            page_content=d["text"],
            metadata={"id": d["id"], "source": d["source"], "title": d["title"]},
        )
        for d in squad_docs
    ]

    emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    chroma_path = _examples_dir / ".chroma_ci_gate"
    store = Chroma.from_documents(
        documents=docs,
        embedding=emb,
        persist_directory=str(chroma_path),
        collection_name="ci_gate",
    )

    llm = FakeListChatModel(responses=[_HEURISTIC_JSON])

    retriever = AdaptiveRAGRetriever(
        vector_store=store,
        llm=llm,
        embedding_model=emb,
        max_attempts=2,
        min_confidence=0.5,
        metrics_collector=MetricsCollector(),
        cost_tracker=CostTracker(),
        enable_logging=False,
    )
    return retriever, emb, store


async def run_gate() -> int:
    """
    Main CI gate logic.
    Returns 0 (pass) or 1 (fail).
    """
    golden_path = _examples_dir / "golden_qa.json"
    docs_registry_path = _examples_dir / "golden_docs_registry.json"
    baselines_path = _examples_dir / "baselines.json"
    dataset_name = "squad_ci"

    print("=" * 55)
    print("  ragfallback MLOps — CI Regression Gate")
    print("=" * 55)

    # ── Check required files ──
    for f in [golden_path, docs_registry_path]:
        if not f.exists():
            print(f"FAIL: {f.name} not found. Run build_golden_dataset.py first.")
            return 1

    # ── Load dataset ──
    samples = json.loads(golden_path.read_text())[:20]  # 20 samples keeps CI fast
    print(f"\n  Dataset   : {golden_path.name} ({len(samples)} samples)")

    # ── Build vector store + retriever ──
    print("  Embedding : all-MiniLM-L6-v2 (local, no API key)")
    print("  Building ChromaDB index...")
    t0 = time.perf_counter()
    retriever, emb, _ = _build_store_and_retriever(docs_registry_path)
    print(f"  Index ready in {(time.perf_counter() - t0) * 1000:.0f}ms")

    # ── Run GoldenRunner ──
    from ragfallback.mlops import GoldenRunner, RagasHook

    hook = RagasHook(llm=None, embeddings=emb)
    runner = GoldenRunner(retriever=retriever, ragas_hook=hook, dataset=samples)

    # Warm up retrieval + embed path so P95 is comparable across runs (Chroma cold start).
    for w in samples[:5]:
        retriever.query_with_fallback(w["query"])

    print("\n  Running GoldenRunner (async)...")
    t0 = time.perf_counter()
    report = await runner.run_async()
    print(f"  Completed in {(time.perf_counter() - t0) * 1000:.0f}ms")

    # ── Print results ──
    print()
    print(f"  {'Metric':<28} {'Value':>8}")
    print(f"  {'─' * 28} {'─' * 8}")
    print(f"  {'recall@3':<28} {report.recall_at_3:>8.3f}")
    print(f"  {'recall@5':<28} {report.recall_at_5:>8.3f}")
    print(f"  {'faithfulness':<28} {report.ragas.faithfulness:>8.3f}")
    print(f"  {'answer_relevance':<28} {report.ragas.answer_relevance:>8.3f}")
    print(f"  {'context_precision':<28} {report.ragas.context_precision:>8.3f}")
    print(f"  {'latency_p95_ms':<28} {report.latency_p95_ms:>8.1f}")
    print(f"  {'fallback_rate':<28} {report.fallback_rate:>8.1%}")
    print(f"  {'ragas_fallback_mode':<28} {str(report.ragas.fallback_mode):>8}")

    # ── Regression gate ──
    from ragfallback.mlops import BaselineRegistry, RegressionError

    registry = BaselineRegistry(str(baselines_path))
    print()

    if not registry.exists(dataset_name):
        registry.update(report, dataset=dataset_name)
        print(f"  FIRST RUN — baseline saved to {baselines_path.name}")
        print("  Commit baselines.json to repo to enable regression gating.")
        print("\n  RESULT: PASS (baseline established)")
        return 0

    # Compare against committed baseline
    baseline = registry.get(dataset_name)
    print(
        f"  Comparing against baseline (recorded: {baseline.get('recorded_at', 'unknown')})"
    )
    print("  Threshold: 5% quality metrics; 12% P95 latency (CI noise) → FAIL")

    try:
        registry.compare_or_fail(
            report,
            dataset=dataset_name,
            threshold=0.05,
            latency_threshold=0.12,
        )
        registry.update(report, dataset=dataset_name)
        print("\n  RESULT: PASS ✓ — No regression detected")
        return 0
    except RegressionError as e:
        print("\n  RESULT: FAIL ✗")
        print(f"  {e}")
        print()
        print("  To reset baseline (if intentional change):")
        print(f"    Delete {baselines_path.name} and re-run.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_gate())
    print()
    sys.exit(exit_code)
