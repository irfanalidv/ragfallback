"""
USE CASE: UC-6 — Adaptive RAG (legal contracts)
================================================
Real problem : High-stakes answers need retries and confidence when clauses conflict.
Goal         : Demonstrate AdaptiveRAGRetriever on real US contract clauses.
Module       : ragfallback.core.AdaptiveRAGRetriever
Vector DB    : FAISS (local)
Dataset      : theatticusproject/cuad-qa (US commercial contracts, CC BY 4.0)
               https://huggingface.co/datasets/theatticusproject/cuad-qa
Install      : pip install ragfallback[faiss,huggingface,real-data]
Env vars     : NONE required
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
    print("ragfallback — Legal Contract RAG Demo")
    print("Dataset: theatticusproject/cuad-qa (CC BY 4.0)")
    print("=" * 70)

    try:
        from datasets import load_dataset
    except ImportError:
        print("SKIP: pip install ragfallback[real-data] to use real legal data")
        sys.exit(0)

    from langchain_core.documents import Document

    print("\nLoading CUAD contract dataset...")
    try:
        ds = load_dataset("theatticusproject/cuad-qa", split="test")
    except Exception as exc:
        print(f"SKIP: Could not load dataset — {exc}")
        sys.exit(0)

    seen: set = set()
    documents: list[Document] = []
    test_question = "What are the termination conditions?"

    for row in ds:
        ctx = (row.get("context") or "").strip()
        if ctx and ctx not in seen and len(ctx) > 100:
            documents.append(Document(
                page_content=ctx[:600],
                metadata={"source": "cuad_contract", "title": str(row.get("title", ""))},
            ))
            seen.add(ctx)
        if row.get("question") and test_question == "What are the termination conditions?":
            test_question = row["question"]
        if len(documents) >= 60:
            break

    if not documents:
        print("SKIP: No usable contract clauses loaded from dataset")
        sys.exit(0)

    print(f"  Loaded {len(documents)} real contract clauses from CUAD")
    print(f"  Example: {documents[0].page_content[:100]}...")
    print(f"  Test question: {test_question[:80]}")

    from ragfallback.diagnostics import ChunkQualityChecker

    checker = ChunkQualityChecker(min_chars=80)
    report = checker.check(documents)
    print(f"\nChunkQualityChecker: {report.n_chunks} clauses  Violations: {len(report.violations)}")

    import _kb_common

    emb = _kb_common.get_embeddings()

    try:
        from langchain_community.vectorstores import FAISS
    except ImportError:
        print("SKIP: pip install ragfallback[faiss] for vector store")
        sys.exit(0)

    print("\nBuilding FAISS index...")
    vs = FAISS.from_documents(documents, emb)
    print(f"  Indexed {len(documents)} real contract clauses")

    print("\nRetrieval demo (no LLM required)...")
    questions = [
        test_question,
        "What are the payment terms?",
        "What happens upon termination?",
    ]
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    for q in questions[:2]:
        hits = retriever.invoke(q)
        best = hits[0].page_content[:90] if hits else "—"
        print(f"  Q: {q[:60]}")
        print(f"     → {best}...")

    print("\n✅ Legal contract RAG demo complete (no paid API keys used).")


if __name__ == "__main__":
    main()
