"""
USE CASE: UC-3 — Chunk quality check before embedding
======================================================
Real problem : PDF and web scrapers create wildly uneven chunks — 10-word stubs,
               mid-sentence cuts, near-duplicate paragraphs. Embedding these
               poisons the index silently: retrieval "succeeds" (returns k docs)
               but the chunks are off-topic noise. No error is ever raised.
Goal         : Run ChunkQualityChecker on 200 real Wikipedia passages from SQuAD,
               identify actual violations, show suggest_fixes() output, and
               demonstrate auto_fix() reducing the violation count.
Dataset      : SQuAD Wikipedia validation set (CC BY-SA 4.0)
               https://huggingface.co/datasets/rajpurkar/squad
Module       : ragfallback.diagnostics.ChunkQualityChecker
Vector DB    : N/A
Install      : pip install ragfallback[real-data]
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
from ragfallback.diagnostics import ChunkQualityChecker  # noqa: E402


def _show_violation_example(texts: list[str], checker: ChunkQualityChecker) -> None:
    """Find and display the first real violation with its diagnosis."""
    for text in texts:
        report = checker.check([text])
        if report.violations:
            words = text.split()
            preview = " ".join(words[:20]) + ("..." if len(words) > 20 else "")
            print(f"\n  Chunk  : '{preview}'")
            print(f"  Length : {len(text)} chars, {len(words)} words")
            for v in report.violations[:2]:
                print(f"  Issue  : {v}")
            return


def main() -> None:
    print("=" * 64)
    print("UC-3: ChunkQualityChecker — real Wikipedia passages (SQuAD)")
    print("=" * 64)

    print("\nLoading 200 real SQuAD Wikipedia passages...")
    try:
        docs, _ = _kb_common.load_real_dataset("squad", max_docs=200)
    except ImportError:
        print("  SKIP: pip install ragfallback[real-data] to use real Wikipedia data")
        print("  Falling back to sample KB...")
        docs = _kb_common.load_sample_kb()

    texts = [d.page_content for d in docs]
    print(f"  Loaded {len(texts)} real passages")
    if texts:
        print(f"  Example: {texts[0][:100]}...")

    print("\nRunning ChunkQualityChecker with production thresholds...")
    checker = ChunkQualityChecker(
        min_chars=80,          # flag anything shorter than a sentence
        max_chars=4000,        # flag unusually long passages
        min_words=8,           # flag stubs
    )
    report = checker.check(texts)
    print(report.summary())
    print(f"  avg chars per passage: {report.avg_length:.0f}")
    print(f"  min chars           : {report.min_length}")
    print(f"  max chars           : {report.max_length}")

    print("\nFirst violation found on real Wikipedia text:")
    if report.violations:
        _show_violation_example(texts, checker)
        print("\n  suggest_fixes() recommendations:")
        fixes = checker.suggest_fixes(report)
        if isinstance(fixes, str):
            fixes = fixes.splitlines()
        for line in fixes[:8]:
            print(f"    {line}")
    else:
        # Wikipedia passages are generally high quality — that's the result
        print(
            "  No violations found — Wikipedia passages are well-formed.\n"
            "  This is a GOOD outcome: the checker confirms your data is clean\n"
            "  before you spend time and money embedding it."
        )
        # Inject a synthetic bad chunk to demonstrate the detection path
        print("\n  Injecting 3 synthetic bad chunks to demonstrate detection...")
        synthetic_bad = texts + [
            "Too short.",                           # 10 chars
            "incomplete sentence that just cuts",   # 34 chars, no period
            texts[0][:50] + " " + texts[0][:50],   # near-duplicate
        ]
        bad_report = checker.check(synthetic_bad)
        print(f"  After injection: {bad_report.summary()}")
        _show_violation_example(synthetic_bad, checker)
        print("\n  suggest_fixes():")
        fixes = checker.suggest_fixes(bad_report)
        if isinstance(fixes, str):
            fixes = fixes.splitlines()
        for line in fixes[:8]:
            print(f"    {line}")

    print("\nauto_fix() — filter and repair chunks automatically...")
    before_count = len(texts)
    fixed_texts = checker.auto_fix(texts)
    after_count = len(fixed_texts)
    print(f"  Before : {before_count} passages")
    print(f"  After  : {after_count} passages")
    if after_count < before_count:
        removed = before_count - after_count
        print(f"  Removed: {removed} chunk(s) that failed quality checks")
    else:
        print("  All passages passed — auto_fix removed nothing (clean data ✓)")

    # Verify the fixed set is clean
    fixed_report = checker.check(fixed_texts)
    if not fixed_report.has_issues:
        print("  Re-check: 0 violations after auto_fix ✓")
    else:
        remaining = len(fixed_report.violations)
        print(f"  Re-check: {remaining} violation(s) remain (auto_fix is conservative by design)")

    print("\n✅  UC-3 demo complete.")
    print("    Dataset: SQuAD (Wikipedia) — CC BY-SA 4.0")
    print("    Run this before embedding to avoid poisoning your RAG index.")


if __name__ == "__main__":
    main()
