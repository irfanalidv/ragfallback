"""
USE CASE: UC-9 — Embedding Domain Probe
=========================================
Real problem : A generic embedding model (all-MiniLM trained on Wikipedia)
               is used to index specialized legal or medical text. The model
               has no concept of domain vocabulary. Similarity scores are
               uniformly low for all queries. Users get random-looking results.
               No error is thrown — it just silently returns wrong answers.
Goal         : Detect BEFORE indexing whether the embedding model is a
               poor fit for your document domain.
Module       : ragfallback.diagnostics.EmbeddingQualityProbe
Install      : pip install ragfallback[huggingface]
Env vars     : NONE required
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if (_repo_root / "ragfallback").is_dir() and str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

_examples_dir = Path(__file__).resolve().parent
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

import _kb_common  # noqa: E402
from ragfallback.diagnostics import EmbeddingQualityProbe  # noqa: E402

# General business text — all-MiniLM was trained on this kind of content
_GENERAL_SNIPPETS = [
    "Employees receive 20 days of paid annual leave per calendar year.",
    "All team members must complete the mandatory security training by Q4.",
    "The company offers flexible working hours between 7am and 7pm.",
    "Performance reviews are conducted twice yearly in June and December.",
    "Travel reimbursements must be submitted within 30 days of the trip.",
]

# Highly specialized medical/legal jargon — far outside the model's training distribution
_SPECIALIZED_SNIPPETS = [
    "Myocardial infarction with ST-segment elevation treated via percutaneous coronary intervention.",
    "Heretofore the indemnifying party shall hold harmless the indemnitee against tortious claims.",
    "Staphylococcus aureus bacteremia with endocarditis; recommend vancomycin MIC ≤ 1 μg/mL.",
    "Force majeure clause supersedes liquidated damages under privity of contract doctrine.",
    "The appellant avers that the promissory estoppel doctrine bars the statute of limitations defense.",
]


def _run_probe(
    probe: EmbeddingQualityProbe,
    embeddings: object,
    label: str,
    query: str,
    snippets: list[str],
) -> None:
    result = probe.run(embeddings=embeddings, query=query, reference_snippets=snippets)  # type: ignore[arg-type]
    ok_str = "OK ✓" if result.ok else "WARN ✗"
    print(f"\n  Domain : {label}")
    print(f"  Query  : '{query}'")
    print(f"  Status : {ok_str}")
    print(f"  mean_top1_similarity = {result.mean_top1_similarity:.3f}")
    print(f"  spread               = {result.spread:.3f}")
    if result.warnings:
        for w in result.warnings:
            print(f"  ⚠  {w}")
    if not result.ok:
        print(
            "  → Action: if ok=False, consider a domain-specific embedding model "
            "or fine-tuning before indexing."
        )


def main() -> None:
    print("=" * 60)
    print("UC-9: EmbeddingQualityProbe demo")
    print("=" * 60)
    print("\nModel: all-MiniLM-L6-v2 (generic, trained on Wikipedia / web text)")

    emb = _kb_common.get_embeddings()
    probe = EmbeddingQualityProbe(min_mean_top1=0.25, max_spread=0.15)

    _run_probe(
        probe, emb,
        label="General business HR text",
        query="What is the company leave policy?",
        snippets=_GENERAL_SNIPPETS,
    )

    _run_probe(
        probe, emb,
        label="Specialized medical/legal jargon",
        query="What is the recommended treatment protocol?",
        snippets=_SPECIALIZED_SNIPPETS,
    )

    print("\n" + "-" * 60)
    print("Interpretation:")
    print(
        "  - ok=True  → model similarity distribution looks healthy for this domain."
    )
    print(
        "  - ok=False → poor domain fit detected. Consider:"
    )
    print("      * domain-specific model (e.g. BioBERT, LegalBERT)")
    print("      * fine-tuning on domain text")
    print("      * hybrid search (BM25 + dense) — see UC-5")


if __name__ == "__main__":
    main()
