"""
USE CASE: UC-10 — Metadata Sanitizer
=======================================
Real problem : Documents loaded from PDFs, databases, or web scrapers often
               have metadata with nested dicts, lists, bytes, or None values.
               ChromaDB and Pinecone reject these silently or throw cryptic
               errors. The entire indexing job fails.
Goal         : Normalize document metadata to JSON-safe scalar values before
               any vector store write.
Module       : ragfallback.diagnostics.sanitize_documents
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
from ragfallback.diagnostics import sanitize_documents  # noqa: E402


def _print_metadata(label: str, docs: list[Document]) -> None:
    print(f"\n{label}:")
    for i, d in enumerate(docs, 1):
        print(f"  Doc {i}: {d.metadata}")


def main() -> None:
    print("=" * 60)
    print("UC-10: Metadata Sanitizer demo")
    print("=" * 60)

    # Three documents with the kinds of metadata that break vector stores
    dirty_docs = [
        Document(
            page_content="Python lists are mutable ordered sequences of elements.",
            metadata={
                "tags": ["python", "data-structures", "lists"],  # list — rejected by Chroma
                "score": 0.95,
                "source": "python_guide.txt",
            },
        ),
        Document(
            page_content="Dictionaries map hashable keys to arbitrary values.",
            metadata={
                "nested": {"author": "Jane Doe", "year": 2023},  # nested dict — rejected
                "raw_bytes": b"binary content here",              # bytes — rejected
                "source": "python_guide.txt",
            },
        ),
        Document(
            page_content="Functions are defined with the def keyword and return values.",
            metadata={
                "page": None,        # None — handled inconsistently across stores
                "reviewed": True,
                "source": "python_guide.txt",
            },
        ),
    ]

    _print_metadata("Before sanitization (dirty metadata)", dirty_docs)

    emb = _kb_common.get_embeddings()
    persist = Path(tempfile.mkdtemp(prefix="ragfallback_uc10_")) / "chroma"
    print("\n--- Attempting to insert dirty docs into ChromaDB ---")
    try:
        vs_dirty = _kb_common.build_chroma_store(
            dirty_docs,
            persist_directory=persist / "dirty",
            collection_name="uc10_dirty",
        )
        print("  ChromaDB accepted the docs (it serialized lists/None silently).")
        print("  NOTE: Pinecone, pgvector, and Qdrant would raise errors here.")
    except Exception as exc:
        print(f"  ChromaDB raised: {type(exc).__name__}: {exc}")
        print("  This is exactly the error sanitize_documents prevents.")

    clean_docs = sanitize_documents(dirty_docs)
    _print_metadata("After sanitization (clean scalar metadata)", clean_docs)

    print("\n--- Inserting sanitized docs into ChromaDB ---")
    vs_clean = _kb_common.build_chroma_store(
        clean_docs,
        persist_directory=persist / "clean",
        collection_name="uc10_clean",
    )
    results = vs_clean.similarity_search("What are Python lists?", k=2)
    print(f"  Inserted {len(clean_docs)} docs. Retrieved {len(results)} result(s). ✓")
    print(f"\n  Metadata sanitized: {len(clean_docs)} document(s) ready for any vector store.")

    print("\nKey transformations applied by sanitize_documents():")
    print("  list / tuple  → JSON string  (e.g. '[\"python\", \"lists\"]')")
    print("  dict          → flattened keys with dot notation  (nested.key → value)")
    print("  bytes         → UTF-8 decoded string")
    print("  None          → preserved as None (most stores handle scalar None)")


if __name__ == "__main__":
    main()
