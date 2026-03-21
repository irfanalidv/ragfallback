"""
USE CASE: UC-6 — Adaptive RAG (financial news)
===============================================
Real problem : Regulatory and risk queries need retrieval + rewrite when phrasing diverges.
Goal         : Demonstrate AdaptiveRAGRetriever on real financial news sentences.
Module       : ragfallback.core.AdaptiveRAGRetriever
Vector DB    : FAISS (local)
Dataset      : nickmuchi/financial-classification (financial news, Apache 2.0)
               https://huggingface.co/datasets/nickmuchi/financial-classification
Install      : pip install ragfallback[faiss,huggingface,real-data]
Env vars     : NONE required for retrieval demo; HF_TOKEN optional for LLM
"""

from __future__ import annotations

import sys
import os

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(_repo_root, "ragfallback")) and _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

_examples_dir = os.path.dirname(os.path.abspath(__file__))
if _examples_dir not in sys.path:
    sys.path.insert(0, _examples_dir)


def main() -> None:
    print("=" * 70)
    print("ragfallback — Financial News RAG Demo")
    print("Dataset: nickmuchi/financial-classification (Apache 2.0)")
    print("=" * 70)

    try:
        from datasets import load_dataset
    except ImportError:
        print("SKIP: pip install ragfallback[real-data] to use real financial data")
        sys.exit(0)

    from langchain_core.documents import Document

    print("\nLoading financial news dataset...")
    try:
        ds = load_dataset("nickmuchi/financial-classification", split="train")
    except Exception as exc:
        print(f"SKIP: Could not load dataset — {exc}")
        sys.exit(0)

    documents = [
        Document(
            page_content=row["sentence"],
            metadata={"source": "financial_news", "label": str(row.get("label", ""))},
        )
        for row in ds
        if row.get("sentence") and len(row["sentence"]) > 50
    ][:80]

    if not documents:
        print("SKIP: No usable sentences loaded from dataset")
        sys.exit(0)

    print(f"  Loaded {len(documents)} real financial news sentences")
    print(f"  Example: {documents[0].page_content[:100]}...")

    from ragfallback.diagnostics import ChunkQualityChecker

    checker = ChunkQualityChecker(min_chars=40)
    report = checker.check(documents)
    print(f"\nChunkQualityChecker: {report.n_chunks} sentences  Violations: {len(report.violations)}")

    import _kb_common

    emb = _kb_common.get_embeddings()

    try:
        from langchain_community.vectorstores import FAISS
    except ImportError:
        print("SKIP: pip install ragfallback[faiss] for vector store")
        sys.exit(0)

    print("\nBuilding FAISS index...")
    vs = FAISS.from_documents(documents, emb)
    print(f"  Indexed {len(documents)} real sentences")

    print("\nRetrieval demo (no LLM required)...")
    questions = [
        "What happened to company earnings?",
        "What is the outlook for revenue growth?",
        "How did the stock perform?",
    ]
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    for q in questions:
        hits = retriever.invoke(q)
        best = hits[0].page_content[:90] if hits else "—"
        print(f"  Q: {q}")
        print(f"     → {best}...")

    print("\n✅ Financial RAG demo complete (no paid API keys used).")
    print("   To add LLM generation: set HF_TOKEN env var and pass an LLM to AdaptiveRAGRetriever.")


if __name__ == "__main__":
    main()
