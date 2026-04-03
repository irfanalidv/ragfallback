"""
Build Golden Dataset for ragfallback MLOps Evaluation
======================================================
Pulls 75 real QA pairs from SQuAD (Wikipedia, CC BY-SA 4.0) and formats
them into golden_qa.json for use with GoldenRunner + BaselineRegistry.

Also pulls 25 from SciQ for a mixed domain stress set (golden_qa_stress.json).

Install : pip install ragfallback[real-data,chroma,huggingface]
Run     : python examples/build_golden_dataset.py
Output  : examples/golden_qa.json          (75 SQuAD samples)
          examples/golden_qa_stress.json   (25 SQuAD + 25 SciQ mixed)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Allow running directly from repo root without pip install -e .
_repo_root = Path(__file__).resolve().parent.parent
if (_repo_root / "ragfallback").is_dir() and str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def _doc_id(text: str, prefix: str = "doc") -> str:
    """Stable deterministic ID from content hash."""
    h = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"{prefix}_{h}"


def build_squad_samples(n: int = 75) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load SQuAD validation split.

    Returns:
        (samples, docs_meta) where samples follow GoldenRunner format:
        {"query", "ground_truth", "relevant_doc_ids"}
        and docs_meta is a list of {"id", "text", "title"} for reference.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("ERROR: pip install ragfallback[real-data]")
        sys.exit(1)

    print("  Downloading SQuAD validation split...")
    ds = load_dataset("rajpurkar/squad", split="validation")

    # Build passage registry: context_text → doc_id
    passage_registry: Dict[str, str] = {}
    samples: List[Dict[str, Any]] = []
    docs_meta: List[Dict[str, Any]] = []

    # We need good samples: has an answer, answer is in context, not too short
    for row in ds:
        if len(samples) >= n:
            break

        context = row["context"].strip()
        question = row["question"].strip()
        answers = row["answers"]["text"]

        if not answers:
            continue
        ground_truth = answers[0].strip()

        # Skip trivial answers (too short to be meaningful)
        if len(ground_truth) < 3:
            continue

        # Register the passage
        if context not in passage_registry:
            doc_id = _doc_id(context, prefix="squad")
            passage_registry[context] = doc_id
            docs_meta.append(
                {
                    "id": doc_id,
                    "text": context,
                    "title": row["title"],
                    "source": "squad",
                }
            )
        else:
            doc_id = passage_registry[context]

        samples.append(
            {
                "query": question,
                "ground_truth": ground_truth,
                "relevant_doc_ids": [doc_id],  # the passage that contains the answer
                "metadata": {
                    "source": "squad",
                    "title": row["title"],
                    "doc_id": doc_id,
                },
            }
        )

    print(f"  SQuAD: {len(samples)} samples, {len(docs_meta)} unique passages")
    return samples, docs_meta


def build_sciq_samples(n: int = 25) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load SciQ test split — science domain, harder than SQuAD.

    Returns same format as build_squad_samples.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("ERROR: pip install ragfallback[real-data]")
        sys.exit(1)

    print("  Downloading SciQ test split...")
    ds = load_dataset("allenai/sciq", split="test")

    samples: List[Dict[str, Any]] = []
    docs_meta: List[Dict[str, Any]] = []

    for row in ds:
        if len(samples) >= n:
            break

        support = (row.get("support") or "").strip()
        question = row["question"].strip()
        answer = row["correct_answer"].strip()

        # SciQ: skip rows with no supporting passage
        if len(support) < 50:
            continue

        doc_id = _doc_id(support, prefix="sciq")
        docs_meta.append(
            {
                "id": doc_id,
                "text": support,
                "title": "SciQ",
                "source": "sciq",
            }
        )

        samples.append(
            {
                "query": question,
                "ground_truth": answer,
                "relevant_doc_ids": [doc_id],
                "metadata": {
                    "source": "sciq",
                    "doc_id": doc_id,
                },
            }
        )

    print(f"  SciQ: {len(samples)} samples, {len(docs_meta)} unique passages")
    return samples, docs_meta


def write_dataset(samples: List[Dict[str, Any]], path: Path) -> None:
    """Write samples to JSON file."""
    # Remove metadata key from final output (GoldenRunner doesn't need it)
    clean = []
    for s in samples:
        clean.append(
            {
                "query": s["query"],
                "ground_truth": s["ground_truth"],
                "relevant_doc_ids": s["relevant_doc_ids"],
            }
        )
    path.write_text(json.dumps(clean, indent=2, ensure_ascii=False))
    print(f"  Written: {path} ({len(clean)} samples)")


def write_docs_registry(docs: List[Dict[str, Any]], path: Path) -> None:
    """Write passage registry — useful for building vector store from same data."""
    path.write_text(json.dumps(docs, indent=2, ensure_ascii=False))
    print(f"  Written: {path} ({len(docs)} passages)")


def main() -> None:
    print("=" * 60)
    print("ragfallback — Build Golden Dataset from Open Data")
    print("=" * 60)

    out_dir = Path(__file__).resolve().parent
    squad_json = out_dir / "golden_qa.json"
    stress_json = out_dir / "golden_qa_stress.json"
    docs_registry = out_dir / "golden_docs_registry.json"

    # --- SQuAD: primary golden dataset ---
    print("\n[1/3] Building primary golden dataset (SQuAD, n=75)...")
    squad_samples, squad_docs = build_squad_samples(n=75)
    write_dataset(squad_samples, squad_json)

    # --- SciQ: stress set ---
    print("\n[2/3] Building stress golden dataset (SciQ, n=25)...")
    sciq_samples, sciq_docs = build_sciq_samples(n=25)

    # Stress set = 25 SQuAD + 25 SciQ (mixed domain)
    stress_samples = squad_samples[:25] + sciq_samples
    write_dataset(stress_samples, stress_json)

    # --- Docs registry ---
    print("\n[3/3] Writing passage registry (for vector store construction)...")
    all_docs = squad_docs + sciq_docs
    write_docs_registry(all_docs, docs_registry)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("DONE. Files written:")
    print(f"  {squad_json.name:<35} — 75 SQuAD samples (primary eval)")
    print(f"  {stress_json.name:<35} — 50 mixed samples (stress eval)")
    print(f"  {docs_registry.name:<35} — passage registry")
    print()
    print("Next step:")
    print("  python examples/mlops_demo.py")
    print()
    print("Licenses:")
    print("  SQuAD : CC BY-SA 4.0  (https://huggingface.co/datasets/rajpurkar/squad)")
    print("  SciQ  : CC BY-NC 3.0  (https://huggingface.co/datasets/allenai/sciq)")
    print("=" * 60)


if __name__ == "__main__":
    main()
