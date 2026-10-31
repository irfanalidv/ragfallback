"""
Ingestion pre-flight: catch bad chunks, unsafe metadata, broken embeddings, and a
stale index *before* anything reaches the vector store.

What it shows
    1. ``ChunkQualityChecker`` flags fragments and oversized chunks from a sloppy
       splitter, and ``auto_fix`` repairs them.
    2. ``sanitize_documents`` flattens list/dict/bytes metadata that Pinecone,
       Qdrant, or pgvector would reject.
    3. ``EmbeddingGuard`` blocks an embedder whose width does not match the index.
    4. ``StaleIndexDetector`` fingerprints the source files at build time and later
       reports which files changed, were removed, or are new.

What it needs
    Nothing beyond ``pip install ragfallback``. Works on a temporary copy of the
    bundled ``examples/sample_kb`` articles; no network, no API key.

Run
    python examples/offline_ingest_preflight.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _offline import SAMPLE_KB, HashingEmbeddings
from langchain_core.documents import Document

from ragfallback.diagnostics import (
    ChunkQualityChecker,
    EmbeddingGuard,
    StaleIndexDetector,
    sanitize_documents,
)


def naive_split(text: str, size: int) -> list[str]:
    """A fixed-width splitter that ignores sentence boundaries (the usual culprit)."""
    return [text[i : i + size] for i in range(0, len(text), size)]


def main() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="ragfallback_preflight_"))
    kb = workdir / "kb"
    shutil.copytree(SAMPLE_KB, kb)
    files = sorted(kb.glob("*.md"))

    print("1) Chunk quality")
    raw = "\n\n".join(f.read_text(encoding="utf-8") for f in files)
    chunks = [*naive_split(raw, 1500), "See above.", "TODO"]
    checker = ChunkQualityChecker(min_chars=80, max_chars=1200)
    report = checker.check(chunks)
    print(f"   before: {report.summary()}")
    for v in report.violations[:4]:
        print(f"     - {v}")
    fixed = checker.auto_fix(chunks, target_chars=900, overlap_chars=100)
    print(f"   after auto_fix: {checker.check(fixed).summary()}")

    print("\n2) Metadata sanitization")
    dirty = Document(
        page_content=fixed[0],
        metadata={"source": files[0].name, "tags": ["billing", "refunds"], "owner": {"team": "cx"}},
    )
    clean = sanitize_documents([dirty])[0]
    print(f"   before: {dirty.metadata}")
    print(f"   after : {clean.metadata}")

    print("\n3) Embedding dimension guard")
    index_dim = 256
    ok = EmbeddingGuard(expected_dim=index_dim).validate(HashingEmbeddings(dim=256))
    bad = EmbeddingGuard(expected_dim=index_dim).validate(HashingEmbeddings(dim=384))
    print(f"   same model : ok={ok.ok}  {ok.message}")
    print(f"   swapped    : ok={bad.ok}  {bad.message[:90]}...")

    print("\n4) Stale index detection")
    detector = StaleIndexDetector(manifest_path=workdir / "index_manifest.json")
    detector.record_paths(files)  # call right after a successful index build
    print(f"   right after build: {detector.check_paths(files).summary()}")

    (kb / "billing_refunds.md").write_text("# Billing\n\nThe refund window is now 45 days.\n")
    (kb / "rate_limits.md").unlink()
    new_file = kb / "sso.md"
    new_file.write_text("# SSO\n\nSAML single sign-on is available on Enterprise plans.\n")
    current = [*sorted(kb.glob("*.md")), kb / "rate_limits.md"]
    stale = detector.check_paths(current)
    print(f"   after doc edits  : {stale.summary()}")
    for label, paths in (
        ("changed", stale.stale_paths),
        ("removed", stale.missing_paths),
        ("new", stale.new_paths),
    ):
        for p in paths:
            print(f"     {label:<8}{Path(p).name}")

    shutil.rmtree(workdir, ignore_errors=True)
    return 0 if not bad.ok and stale.has_stale else 1


if __name__ == "__main__":
    raise SystemExit(main())
