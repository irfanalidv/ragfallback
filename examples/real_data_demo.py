"""
USE CASE: UC-1 through UC-7 — Real Data End-to-End Demo
=========================================================
Real problem : All other examples use text we wrote ourselves.
               This proves ragfallback works on real public data
               with ground-truth answers anyone can verify.
Dataset      : SQuAD validation set — Wikipedia passages + Q&A pairs
               License: CC BY-SA 4.0
               https://huggingface.co/datasets/rajpurkar/squad
Goal         : Run every ragfallback module on 200 real Wikipedia
               passages and show concrete metrics with real numbers.
Vector DB    : ChromaDB (local, no server needed)
Install      : pip install ragfallback[chroma,huggingface,real-data]
Env vars     : NONE required
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Allow running directly from a repository clone without pip install -e .
_repo_root = Path(__file__).resolve().parent.parent
if (_repo_root / "ragfallback").is_dir() and str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

_examples_dir = Path(__file__).resolve().parent
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

import _kb_common  # noqa: E402


def main() -> None:
    print("=" * 65)
    print("ragfallback — Real Data Demo (SQuAD Wikipedia dataset)")
    print("=" * 65)

    print("\nLoading SQuAD dataset (200 Wikipedia passages)...")
    try:
        docs, probes = _kb_common.load_real_dataset("squad", max_docs=200)
    except ImportError as exc:
        print(f"SKIP: {exc}")
        sys.exit(0)

    print(f"  Passages loaded : {len(docs)}")
    print(f"  Q&A pairs       : {len(probes)}")
    print(f"  Example passage : {docs[0].page_content[:120]}...")
    print(f"  Example question: {probes[0]['question']}")
    print(f"  Ground truth    : {probes[0]['ground_truth']}")

    print("\nChunkQualityChecker on real Wikipedia passages...")
    from ragfallback.diagnostics import ChunkQualityChecker

    chunk_report = ChunkQualityChecker().check(docs)
    violation_count = len(chunk_report.violations)
    print(f"  Total passages  : {chunk_report.n_chunks}")
    print(f"  Violations      : {violation_count}")
    print(f"  Avg chars       : {chunk_report.avg_length:.0f}")
    print(f"  Min/max chars   : {chunk_report.min_length} / {chunk_report.max_length}")
    if chunk_report.violations:
        print(f"  First violation : {chunk_report.violations[0]}")
    else:
        print("  Result          : all chunks within spec")

    print("\nEmbeddingGuard validation...")
    from ragfallback.diagnostics import EmbeddingGuard

    embeddings_model = _kb_common.get_embeddings()
    guard = EmbeddingGuard(expected_dim=384)
    val_result = guard.validate(embeddings_model)
    print(f"  Status : {'OK' if val_result.ok else 'FAIL'}")
    print(f"  Message: {val_result.message}")

    chroma_path = Path(os.path.dirname(__file__)) / ".chroma_squad_demo"
    print(f"\nBuilding ChromaDB index ({len(docs)} real Wikipedia passages)...")
    store = _kb_common.build_chroma_store(
        docs,
        persist_directory=chroma_path,
        collection_name="squad_wikipedia",
    )
    print(f"  Index built at  : {chroma_path}")

    print("\nRetrievalHealthCheck (20 real Q&A substring probes)...")
    from ragfallback.diagnostics import RetrievalHealthCheck

    # SQuAD answers are spans within context, so substring match is the correct check
    probe_dict = {
        p["question"]: p["ground_truth"][:60]
        for p in probes[:20]
    }
    health_checker = RetrievalHealthCheck(k=4)
    health_report = health_checker.run_substring_probes(
        store,
        probe_dict,
        min_mean_hit_rate=0.30,  # realistic for embedding model on Wikipedia spans
    )
    print(f"  Hit rate        : {health_report.hit_rate:.1%}")
    print(f"  Healthy (>=30%) : {'YES' if health_report.ok else 'NO'}")
    if health_report.avg_latency_ms is not None:
        print(f"  Avg latency     : {health_report.avg_latency_ms:.0f}ms per query")

    missed = [r["query"] for r in health_report.per_query if r.get("recall", 0) == 0]
    if missed:
        print(f"  Missed ({len(missed)}/{len(probe_dict)}): e.g. '{missed[0][:55]}...'")

    print("\nRAGEvaluator on 10 real Wikipedia Q&A pairs...")
    from ragfallback.evaluation import RAGEvaluator

    evaluator = RAGEvaluator()
    scores = []
    retriever = store.as_retriever(search_kwargs={"k": 4})
    for probe in probes[:10]:
        retrieved = retriever.invoke(probe["question"])
        contexts = [d.page_content for d in retrieved]
        # First retrieved context is the simulated answer (extractive RAG)
        answer = contexts[0] if contexts else "No answer found."
        score = evaluator.evaluate(
            question=probe["question"],
            answer=answer,
            contexts=contexts,
            ground_truth=probe["ground_truth"],
        )
        scores.append(score)

    pass_count = sum(1 for s in scores if s.passed)
    avg_recall = sum((s.recall_at_k or 0) for s in scores) / len(scores)
    avg_faith = sum(s.faithfulness_score for s in scores) / len(scores)
    avg_overall = sum(s.overall_score for s in scores) / len(scores)

    print(f"  Questions       : {len(scores)}")
    print(f"  Pass rate       : {pass_count}/{len(scores)}")
    print(f"  Avg recall@k    : {avg_recall:.1%}")
    print(f"  Avg faithfulness: {avg_faith:.1%}")
    print(f"  Avg overall     : {avg_overall:.1%}")
    print(f"\n  {evaluator.batch_summary(scores)}")

    print("\n" + "=" * 65)
    print("Real data demo complete.")
    print("Dataset: SQuAD (Wikipedia) — CC BY-SA 4.0")
    print("Source : https://huggingface.co/datasets/rajpurkar/squad")


if __name__ == "__main__":
    main()
