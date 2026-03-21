"""
Shared KB loading and vector-store builders for examples.

No imports from ``ragfallback`` — only LangChain + vector-store libraries
(see project dev instructions: factories/helpers stay dependency-light here).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

from langchain_core.documents import Document


def _default_kb_dir() -> Path:
    return Path(__file__).resolve().parent / "sample_kb"


def _make_splitter(chunk_size: int, chunk_overlap: int):
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except ImportError:  # pragma: no cover - optional package layout
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter

            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        except ImportError:
            return None


def load_sample_kb(
    sources: Optional[Sequence[str]] = None,
    *,
    chunk_size: int = 400,
    chunk_overlap: int = 60,
    kb_dir: Optional[Path] = None,
) -> List[Document]:
    """Load ``examples/sample_kb`` markdown files and split into chunks.

    Args:
        sources: If set, only include these basenames (e.g. ``(\"billing_refunds.md\",)``).
        chunk_size: Target chunk size in characters for the text splitter.
        chunk_overlap: Overlap between consecutive chunks.
        kb_dir: Override KB directory (default: ``examples/sample_kb``).

    Returns:
        Chunked LangChain documents with ``source`` metadata.
    """
    root = kb_dir or _default_kb_dir()
    if not root.is_dir():
        raise FileNotFoundError(f"KB directory not found: {root}")

    raw_docs: List[Document] = []
    for path in sorted(root.glob("*.md")):
        if sources is not None and path.name not in sources:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        raw_docs.append(
            Document(
                page_content=text,
                metadata={"source": path.name, "path": str(path.resolve())},
            )
        )

    splitter = _make_splitter(chunk_size, chunk_overlap)
    if splitter is None:
        # Character fallback if text_splitters is unavailable
        out: List[Document] = []
        for d in raw_docs:
            text = d.page_content
            meta = dict(d.metadata)
            for i in range(0, len(text), max(1, chunk_size - chunk_overlap)):
                chunk = text[i : i + chunk_size]
                if chunk.strip():
                    out.append(Document(page_content=chunk, metadata=meta))
        return out

    chunks: List[Document] = []
    for d in raw_docs:
        chunks.extend(splitter.split_documents([d]))
    return chunks


def get_embeddings(model_name: str = "all-MiniLM-L6-v2"):
    """Local HuggingFace sentence-transformers embeddings (no API key)."""
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=model_name)


def build_chroma_store(
    docs: List[Document],
    persist_directory: Union[str, Path],
    collection_name: str = "ragfallback_examples",
):
    """Build or refresh a persisted Chroma collection."""
    from langchain_community.vectorstores import Chroma

    emb = get_embeddings()
    return Chroma.from_documents(
        documents=docs,
        embedding=emb,
        persist_directory=str(persist_directory),
        collection_name=collection_name,
    )


def build_faiss_store(docs: List[Document], embeddings=None):
    """In-memory FAISS index from documents."""
    from langchain_community.vectorstores import FAISS

    emb = embeddings if embeddings is not None else get_embeddings()
    return FAISS.from_documents(documents=docs, embedding=emb)


def load_real_dataset(
    dataset_name: str = "squad",
    max_docs: int = 200,
) -> tuple:
    """Load a real public HuggingFace dataset as LangChain Documents.

    Args:
        dataset_name: ``"squad"`` (Wikipedia Q&A) or ``"sciq"`` (science Q&A).
        max_docs: Maximum number of unique passage documents to return.

    Returns:
        Tuple of ``(docs, probes)`` where *docs* are LangChain Documents and
        *probes* are dicts with ``"question"`` and ``"ground_truth"`` keys.

    Raises:
        ImportError: If the ``datasets`` package is not installed.
        ValueError: If *dataset_name* is not supported.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Real dataset loading requires: pip install ragfallback[real-data]\n"
            "Or: pip install datasets"
        ) from exc

    if dataset_name == "squad":
        ds = load_dataset("rajpurkar/squad", split="validation")
        seen_contexts: dict = {}
        probes: List[dict] = []
        for row in ds:
            ctx = row["context"].strip()
            if ctx not in seen_contexts and len(seen_contexts) < max_docs:
                seen_contexts[ctx] = Document(
                    page_content=ctx,
                    metadata={"source": "squad", "title": row["title"]},
                )
            if row["answers"]["text"]:
                probes.append(
                    {
                        "question": row["question"],
                        "ground_truth": row["answers"]["text"][0],
                    }
                )
        return list(seen_contexts.values()), probes

    elif dataset_name == "sciq":
        ds = load_dataset("allenai/sciq", split="test")
        docs: List[Document] = []
        probes = []
        for row in ds:
            support = (row.get("support") or "").strip()
            if support and len(support) >= 50:
                docs.append(
                    Document(
                        page_content=support,
                        metadata={"source": "sciq"},
                    )
                )
                probes.append(
                    {
                        "question": row["question"],
                        "ground_truth": row["correct_answer"],
                    }
                )
            if len(docs) >= max_docs:
                break
        return docs, probes

    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Supported: 'squad', 'sciq'"
        )
