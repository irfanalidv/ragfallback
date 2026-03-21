"""
USE CASE: UC-8 — Context Stitcher
===================================
Real problem : Retrieved chunks from the same document get split across
               top-k results. The answer spans two adjacent chunks but
               the LLM only sees one half. Retrieval "succeeds" but
               generation fails silently.
Goal         : Merge adjacent chunks from the same source before prompting
               so the LLM always sees complete context.
Module       : ragfallback.diagnostics.OverlappingContextStitcher
Vector DB    : ChromaDB (local, no server)
Install      : pip install ragfallback[chroma,huggingface]
Env vars     : NONE required
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

from langchain_core.documents import Document  # noqa: E402
import _kb_common  # noqa: E402
from ragfallback.diagnostics import OverlappingContextStitcher  # noqa: E402


# Six chunks from the same HR policy document — split across top-k retrieval
_HR_CHUNKS = [
    "The company offers 20 days of paid annual leave per year for all full-time employees.",
    "Annual leave must be approved by your line manager at least two weeks in advance.",
    "Unused leave of up to 5 days may be carried over to the next calendar year.",
    "Sick leave is separate from annual leave and requires a medical certificate after 3 days.",
    "Employees on probation receive 10 days of annual leave in their first year.",
    "Public holidays are in addition to the annual leave entitlement described above.",
]


def _make_hr_docs() -> list[Document]:
    return [
        Document(
            page_content=text,
            metadata={"source": "company_hr.txt", "chunk_index": i},
        )
        for i, text in enumerate(_HR_CHUNKS)
    ]


def main() -> None:
    print("=" * 60)
    print("UC-8: OverlappingContextStitcher demo")
    print("=" * 60)

    docs = _make_hr_docs()
    emb = _kb_common.get_embeddings()

    # Dense retrieval skips chunk 1 (adjacent to chunk 0) and returns
    # non-consecutive indices, fragmenting the answer across top-k.
    persist = Path(tempfile.mkdtemp(prefix="ragfallback_uc8_")) / "chroma"
    vs = _kb_common.build_chroma_store(docs, persist_directory=persist, collection_name="uc8")

    query = "How many days of annual leave do employees get?"
    retrieved = vs.similarity_search(query, k=3)

    print(f"\nQuery: '{query}'")
    print(f"\n--- Before stitching: {len(retrieved)} fragments ---")
    for d in retrieved:
        idx = d.metadata.get("chunk_index", "?")
        print(f"  [chunk {idx}] {d.page_content[:70]}...")

    stitcher = OverlappingContextStitcher(chunk_index_key="chunk_index", source_key="source")
    stitched = stitcher.stitch(retrieved)

    print(f"\n--- After stitching: {len(stitched)} merged context block(s) ---")
    for i, d in enumerate(stitched, 1):
        start = d.metadata.get("stitched_chunk_start", "?")
        end = d.metadata.get("stitched_chunk_end", "?")
        was_stitched = d.metadata.get("ragfallback_stitched", False)
        label = f"chunks {start}–{end}" if was_stitched else f"chunk {start}"
        print(f"\n  Block {i} ({label}, {len(d.page_content)} chars):")
        print(f"  {d.page_content[:200]}{'...' if len(d.page_content) > 200 else ''}")

    print(
        f"\nResult: {len(retrieved)} fragment(s) → {len(stitched)} merged block(s). "
        "The LLM now sees complete, unbroken context."
    )

    print("\n--- expand_neighbor_indices (radius=1) ---")
    expanded = stitcher.expand_neighbor_indices(retrieved[:1], neighbor_radius=1)
    print(f"  Input: 1 chunk → expanded to {len(expanded)} chunk(s) (neighbors included)")


if __name__ == "__main__":
    main()
