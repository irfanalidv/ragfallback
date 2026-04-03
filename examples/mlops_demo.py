"""
ragfallback MLOps Demo — Full End-to-End Pipeline
==================================================
What this demonstrates (and what you explain in interviews):

  1. GOLDEN DATASET     — real SQuAD data, not synthetic
  2. VECTOR STORE       — ChromaDB, local, no API keys
  3. GOLDEN RUNNER      — runs all queries, tracks latency + fallback rate
  4. RAGAS EVALUATION   — faithfulness, answer relevance, context precision
                          (falls back to heuristics if ragas not installed)
  5. BASELINE REGISTRY  — first run saves baseline; subsequent runs gate on it
  6. REGRESSION GATE    — raises RegressionError if any metric degrades > 5%
  7. MLFLOW LOGGING     — logs every run (no-op if mlflow not installed)
  8. QUERY SIMULATION   — adversarial query types from the same dataset
  9. LOCUST FILE        — generates ready-to-run load test

Install : pip install ragfallback[chroma,huggingface,real-data,mlops]
Run     : python examples/mlops_demo.py

No API keys required. All models are local (all-MiniLM-L6-v2).
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import List

# Allow running directly from repo root without pip install -e .
_repo_root = Path(__file__).resolve().parent.parent
if (_repo_root / "ragfallback").is_dir() and str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

_examples_dir = Path(__file__).resolve().parent
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

_HEURISTIC_JSON = '{"answer": "See retrieved context.", "source": "retriever"}'


# ─── SECTION PRINTER ──────────────────────────────────────────────────────────


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def info(msg: str) -> None:
    print(f"    {msg}")


# ─── STEP 1: CHECK GOLDEN DATASET ─────────────────────────────────────────────


def load_golden_dataset(path: Path) -> List[dict]:
    section("STEP 1 — Load golden dataset")
    if not path.exists():
        print(f"\n  golden_qa.json not found at: {path}")
        print("  Run first: python examples/build_golden_dataset.py")
        sys.exit(1)

    samples = json.loads(path.read_text())
    # Use 20 samples for the demo (fast, no GPU needed)
    samples = samples[:20]
    ok(f"Loaded {len(samples)} samples from {path.name}")
    info(f"Example query    : {samples[0]['query']}")
    info(f"Example answer   : {samples[0]['ground_truth']}")
    info(f"Relevant doc IDs : {samples[0]['relevant_doc_ids']}")
    return samples


# ─── STEP 2: BUILD VECTOR STORE FROM THE SAME PASSAGES ───────────────────────


def build_vector_store(docs_registry_path: Path):
    section("STEP 2 — Build ChromaDB from golden passages")

    if not docs_registry_path.exists():
        print(f"  docs registry not found: {docs_registry_path}")
        print("  Run first: python examples/build_golden_dataset.py")
        sys.exit(1)

    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document

    # Load passage registry
    docs_raw = json.loads(docs_registry_path.read_text())
    # Only take squad docs for this demo (faster index build)
    squad_raw = [d for d in docs_raw if d["source"] == "squad"][:50]

    docs = [
        Document(
            page_content=d["text"],
            metadata={"id": d["id"], "source": d["source"], "title": d["title"]},
        )
        for d in squad_raw
    ]

    print(f"  Building index for {len(docs)} passages (all-MiniLM-L6-v2)...")
    t0 = time.perf_counter()
    emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    chroma_path = _examples_dir / ".chroma_mlops_demo"
    store = Chroma.from_documents(
        documents=docs,
        embedding=emb,
        persist_directory=str(chroma_path),
        collection_name="mlops_demo",
    )
    elapsed = (time.perf_counter() - t0) * 1000
    ok(f"ChromaDB built in {elapsed:.0f}ms — {len(docs)} passages indexed")
    info(f"Persist path: {chroma_path}")
    return store, emb


# ─── STEP 3: BUILD RETRIEVER ──────────────────────────────────────────────────


def build_retriever(store, emb):
    section("STEP 3 — Build AdaptiveRAGRetriever (no LLM, heuristic mode)")

    # FakeListChatModel: deterministic JSON answer, no API key (LangChain test utility).
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from ragfallback import AdaptiveRAGRetriever, CostTracker, MetricsCollector

    llm = FakeListChatModel(responses=[_HEURISTIC_JSON])

    retriever = AdaptiveRAGRetriever(
        vector_store=store,
        llm=llm,
        embedding_model=emb,
        max_attempts=2,
        min_confidence=0.5,
        metrics_collector=MetricsCollector(),
        cost_tracker=CostTracker(),
    )
    ok("AdaptiveRAGRetriever ready (FakeListChatModel, no API key needed)")
    return retriever


# ─── STEP 4: RUN GOLDEN RUNNER ────────────────────────────────────────────────


async def run_golden_runner(retriever, samples: List[dict], emb):
    section("STEP 4 — GoldenRunner (async, P95 latency + RAGAS eval)")

    from ragfallback.mlops import GoldenRunner, RagasHook

    # RagasHook: will fall back to heuristics if ragas not installed.
    # No LLM passed → pure heuristic scoring (fast, no API cost).
    hook = RagasHook(llm=None, embeddings=emb)

    runner = GoldenRunner(
        retriever=retriever,
        ragas_hook=hook,
        dataset=samples,
    )

    for w in samples[:5]:
        retriever.query_with_fallback(w["query"])

    print(f"  Running {len(samples)} queries async...")
    t0 = time.perf_counter()
    report = await runner.run_async()
    elapsed = (time.perf_counter() - t0) * 1000

    ok(f"GoldenRunner complete in {elapsed:.0f}ms total")
    print()
    print(f"  {'Metric':<28} {'Value'}")
    print(f"  {'─' * 28} {'─' * 10}")
    print(f"  {'Recall@3':<28} {report.recall_at_3:.3f}")
    print(f"  {'Recall@5':<28} {report.recall_at_5:.3f}")
    print(f"  {'Faithfulness':<28} {report.ragas.faithfulness:.3f}")
    print(f"  {'Answer Relevance':<28} {report.ragas.answer_relevance:.3f}")
    print(f"  {'Context Precision':<28} {report.ragas.context_precision:.3f}")
    print(f"  {'Context Recall':<28} {report.ragas.context_recall:.3f}")
    print(f"  {'Latency P95 (ms)':<28} {report.latency_p95_ms:.1f}")
    print(f"  {'Latency Mean (ms)':<28} {report.latency_mean_ms:.1f}")
    print(f"  {'Fallback Rate':<28} {report.fallback_rate:.1%}")
    print(f"  {'Samples':<28} {report.n_samples}")
    print(f"  {'RAGAS fallback mode':<28} {report.ragas.fallback_mode}")

    return report


# ─── STEP 5: BASELINE REGISTRY ────────────────────────────────────────────────


def run_baseline_registry(report):
    section("STEP 5 — BaselineRegistry (CI regression gate)")

    from ragfallback.mlops import BaselineRegistry, RegressionError

    registry_path = _examples_dir / "baselines_demo.json"
    registry = BaselineRegistry(str(registry_path))

    dataset_name = "squad_20"

    if not registry.exists(dataset_name):
        # First run: save baseline, nothing to compare against
        registry.update(report, dataset=dataset_name)
        ok(f"First run — baseline saved to {registry_path.name}")
        info("Subsequent runs will compare against this baseline.")
        info("Threshold: 5% degradation in any metric triggers RegressionError")
    else:
        # Subsequent runs: compare and gate
        info(f"Existing baseline found for '{dataset_name}'")
        baseline = registry.get(dataset_name)
        info(f"Baseline faithfulness : {baseline.get('faithfulness', 'N/A')}")
        info(f"Current  faithfulness : {report.ragas.faithfulness:.3f}")
        try:
            registry.compare_or_fail(report, dataset=dataset_name, threshold=0.05)
            ok("Regression gate PASSED — no metric degraded > 5%")
            registry.update(report, dataset=dataset_name)
            ok("Baseline updated with current run scores")
        except RegressionError as e:
            print("\n  ✗ REGRESSION GATE FAILED:")
            print(f"    {e}")
            print("  (In CI this would fail the pipeline)")


# ─── STEP 6: MLFLOW LOGGING ───────────────────────────────────────────────────


def run_mlflow_logging(report):
    section("STEP 6 — MLflow logging (no-op if mlflow not installed)")

    from ragfallback.mlops import MLflowLogger

    logger = MLflowLogger(tracking_uri=f"file:{_examples_dir}/mlruns")
    logger.log_golden_report(
        report,
        run_name="ragfallback-mlops-demo",
        dataset="squad_20",
    )
    ok("MLflow log_golden_report called")
    info("If mlflow installed: mlflow ui --backend-store-uri file:./examples/mlruns")
    info("If not installed: no-op (warning logged, no crash)")


# ─── STEP 7: QUERY SIMULATION ─────────────────────────────────────────────────


def run_query_simulation(samples: List[dict]):
    section("STEP 7 — QuerySimulator (adversarial query mix)")

    from ragfallback.mlops import QuerySimulator

    sim = QuerySimulator()
    base_queries = [s["query"] for s in samples[:8]]

    # Standard mix (25% each type)
    mixed = sim.simulate(base_queries)

    type_counts: dict = {}
    for q in mixed:
        type_counts[q.query_type] = type_counts.get(q.query_type, 0) + 1

    ok(f"simulate() → {len(mixed)} queries from {len(base_queries)} originals")
    for qtype, count in sorted(type_counts.items()):
        print(f"    {qtype:<20} {count} queries")

    # Show a sample of each type
    print()
    info("Sample transforms:")
    seen_types: set = set()
    for q in mixed:
        if q.query_type not in seen_types:
            print(f"    [{q.query_type}]")
            print(f"      original  : {q.original[:60]}")
            print(f"      transformed: {q.transformed[:60]}")
            seen_types.add(q.query_type)
        if len(seen_types) == 4:
            break

    # Unhappy paths (4x expansion)
    unhappy = sim.simulate_unhappy_paths(base_queries[:3])
    ok(f"simulate_unhappy_paths() → {len(unhappy)} queries (4× of 3 inputs)")


# ─── STEP 8: LOCUST FILE GENERATION ──────────────────────────────────────────


def run_locust_generation():
    section("STEP 8 — generate_locustfile (Locust load test)")

    from ragfallback.mlops import generate_locustfile

    locust_path = str(_examples_dir / "generated_locustfile.py")
    generate_locustfile(locust_path, endpoint="http://localhost:8000")
    ok(f"Locust file written: {Path(locust_path).name}")
    info("Run load test with:")
    info("  pip install locust")
    info(f"  locust -f {Path(locust_path).name} --host http://localhost:8000")
    info("  Open http://localhost:8089 — set users=50, spawn_rate=5")


# ─── MAIN ─────────────────────────────────────────────────────────────────────


async def main() -> None:
    print("=" * 60)
    print("  ragfallback — MLOps Demo (zero API keys)")
    print("  SQuAD dataset (CC BY-SA 4.0)")
    print("=" * 60)

    golden_path = _examples_dir / "golden_qa.json"
    docs_registry_path = _examples_dir / "golden_docs_registry.json"

    samples = load_golden_dataset(golden_path)
    store, emb = build_vector_store(docs_registry_path)
    retriever = build_retriever(store, emb)
    report = await run_golden_runner(retriever, samples, emb)
    run_baseline_registry(report)
    run_mlflow_logging(report)
    run_query_simulation(samples)
    run_locust_generation()

    section("COMPLETE")
    print("  Full MLOps pipeline ran successfully.")
    print()
    print("  What just happened (interview explanation):")
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │ 1. Golden dataset loaded from SQuAD (open data)     │")
    print("  │ 2. Passages indexed in ChromaDB (local vector store) │")
    print("  │ 3. AdaptiveRAGRetriever queried for all 20 samples  │")
    print("  │ 4. RAGAS metrics computed (or heuristic fallback)   │")
    print("  │ 5. BaselineRegistry gated on 5% regression          │")
    print("  │ 6. MLflow logged all metrics as a tracked run       │")
    print("  │ 7. QuerySimulator generated adversarial query mix   │")
    print("  │ 8. Locust file generated for load testing           │")
    print("  └─────────────────────────────────────────────────────┘")
    print()
    print("  Demo baselines: examples/baselines_demo.json (2nd run hits gate).")
    print("  CI baselines:    examples/baselines.json (see ci_regression_gate.py).")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
