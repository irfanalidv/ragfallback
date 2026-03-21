"""
USE CASE: UC-7 — RAG answer quality evaluation without RAGAS
=============================================================
Real problem : LLM answers sound confident but ignore context or hallucinate.
               RAGAS and TruLens require heavy installs and cloud accounts.
               You need a lightweight quality signal in production right now.
Goal         : Score answer faithfulness, context precision, and recall on
               real Q&A pairs from PubMedQA — medical questions with ground-
               truth answers, where hallucination is genuinely dangerous.
Dataset      : qiaojin/PubMedQA (pqa_labeled) — MIT license
               https://huggingface.co/datasets/qiaojin/PubMedQA
Module       : ragfallback.evaluation.RAGEvaluator
Vector DB    : ChromaDB (local, no server)
Install      : pip install ragfallback[chroma,huggingface,real-data]
Env vars     : NONE required (heuristic mode — no LLM judge needed)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if (_repo_root / "ragfallback").is_dir() and str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

_examples_dir = Path(__file__).resolve().parent
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

import _kb_common  # noqa: E402
from ragfallback.evaluation import RAGEvaluator  # noqa: E402

_N_DOCS = 60
_N_EVAL = 10


def main() -> None:
    print("=" * 62)
    print("UC-7: RAGEvaluator — real medical Q&A (PubMedQA, MIT)")
    print("=" * 62)

    print(f"\nLoading PubMedQA dataset ({_N_DOCS} abstract segments)...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("SKIP: pip install ragfallback[real-data] to use real medical data")
        sys.exit(0)

    from langchain_core.documents import Document

    try:
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    except Exception as exc:
        print(f"SKIP: Could not load dataset — {exc}")
        sys.exit(0)

    docs: list[Document] = []
    probes: list[dict] = []

    for row in list(ds):
        # PubMedQA fields: question, context (dict with "contexts" list), long_answer
        ctx_list = (row.get("context") or {}).get("contexts") or []
        contexts_for_row: list[str] = []
        for ctx in ctx_list:
            if ctx and len(ctx) > 80:
                docs.append(Document(page_content=ctx[:600], metadata={"source": "pubmed"}))
                contexts_for_row.append(ctx[:600])
        if row.get("question") and row.get("long_answer") and contexts_for_row:
            probes.append({
                "question": row["question"],
                "ground_truth": (row.get("long_answer") or "").strip()[:200],
                "contexts": contexts_for_row,
            })
        if len(docs) >= _N_DOCS or len(probes) >= _N_EVAL * 2:
            break

    docs = docs[:_N_DOCS]
    probes = probes[:_N_EVAL]

    if not docs or not probes:
        print("SKIP: No usable data loaded from PubMedQA")
        sys.exit(0)

    print(f"  Abstract segments: {len(docs)}")
    print(f"  Q&A pairs        : {len(probes)}")
    print(f"  Example question : {probes[0]['question'][:80]}...")

    print("\nBuilding ChromaDB index from PubMed abstracts...")
    emb = _kb_common.get_embeddings()
    persist = Path(tempfile.mkdtemp(prefix="ragfallback_uc7_")) / "chroma"
    vs = _kb_common.build_chroma_store(docs, persist_directory=persist, collection_name="uc7")
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    print(f"  Indexed {len(docs)} abstract segments")

    print(f"\nEvaluating {len(probes)} real medical questions (heuristic, no LLM)...")
    ev = RAGEvaluator(llm=None)
    scores = []

    for i, probe in enumerate(probes, 1):
        retrieved = retriever.invoke(probe["question"])
        contexts = [d.page_content for d in retrieved]
        # Extractive RAG: use the best-retrieved context as the answer.
        # This tests whether retrieval found the relevant abstract.
        answer = contexts[0] if contexts else "No answer found."
        score = ev.evaluate(
            question=probe["question"],
            answer=answer,
            contexts=contexts,
            ground_truth=probe["ground_truth"],
        )
        scores.append((probe, score))
        status = "✓" if score.passed else "✗"
        print(
            f"  [{i:2d}] {status}  overall={score.overall_score:.0%}"
            f"  faith={score.faithfulness_score:.0%}"
            f"  recall={score.recall_at_k or 0:.0%}"
            f"  | {probe['question'][:55]}..."
        )

    print("\nBatch summary and failure analysis...")
    all_scores = [s for _, s in scores]
    passed = sum(1 for s in all_scores if s.passed)
    avg_recall = sum(s.recall_at_k or 0 for s in all_scores) / len(all_scores)
    avg_faith = sum(s.faithfulness_score for s in all_scores) / len(all_scores)
    avg_overall = sum(s.overall_score for s in all_scores) / len(all_scores)

    print(f"\n  {passed}/{len(all_scores)} answers passed quality threshold (overall >= 70%)")
    print(f"  Avg recall@k     : {avg_recall:.1%}")
    print(f"  Avg faithfulness : {avg_faith:.1%}")
    print(f"  Avg overall      : {avg_overall:.1%}")

    print("\n" + ev.batch_summary(all_scores))

    failing = [(p, s) for p, s in scores if not s.passed]
    if failing:
        probe, score = failing[0]
        print("\n── Failure analysis (first failing example) ──────────────────")
        print(f"  Question  : {probe['question'][:80]}...")
        print(f"  Answer    : {score.answer[:100]}...")
        print(f"  Ground truth: {probe['ground_truth'][:80]}...")
        print(f"  overall_score = {score.overall_score:.0%} (below 70% threshold)")
        metrics = {
            "context_precision": score.context_precision,
            "faithfulness": score.faithfulness_score,
            "answer_relevance": score.answer_relevance,
        }
        weakest = min(metrics, key=metrics.get)
        print(f"  Weakest metric : {weakest} = {metrics[weakest]:.0%}")
        if metrics[weakest] < 0.4:
            explanations = {
                "context_precision": (
                    "Retrieved contexts don't overlap with the answer — "
                    "retrieval returned off-topic passages."
                ),
                "faithfulness": (
                    "Answer text is not grounded in the retrieved context — "
                    "the extractive answer drifted from what was retrieved."
                ),
                "answer_relevance": (
                    "Answer doesn't directly address the question — "
                    "the retrieved passage answers a related but different question."
                ),
            }
            print(f"  Diagnosis      : {explanations[weakest]}")
        print("  → Action: improve retrieval (SmartThresholdHybridRetriever) or use LLM judge")
    else:
        print("\n  All examples passed! Try stricter thresholds or a harder dataset.")

    print("\n✅  UC-7 demo complete.")
    print("    Dataset: PubMedQA (MIT) — https://huggingface.co/datasets/qiaojin/PubMedQA")
    print("    For higher accuracy: pass llm=your_llm to RAGEvaluator() to enable LLM judge.")


if __name__ == "__main__":
    main()
