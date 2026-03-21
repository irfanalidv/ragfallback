"""
USE CASE: UC-6 — Adaptive RAG (medical abstracts)
==================================================
Real problem : Conflicting abstracts need evidence-based retrieval + confidence scoring.
Goal         : Demonstrate AdaptiveRAGRetriever on real PubMed abstract segments.
Module       : ragfallback.core.AdaptiveRAGRetriever
Vector DB    : FAISS (local)
Dataset      : qiaojin/PubMedQA pqa_labeled split (MIT license)
               https://huggingface.co/datasets/qiaojin/PubMedQA
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
    print("ragfallback — Medical Abstract RAG Demo")
    print("Dataset: qiaojin/PubMedQA pqa_labeled (MIT license)")
    print("=" * 70)

    try:
        from datasets import load_dataset
    except ImportError:
        print("SKIP: pip install ragfallback[real-data] to use real medical data")
        sys.exit(0)

    from langchain_core.documents import Document

    print("\nLoading PubMedQA dataset...")
    try:
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    except Exception as exc:
        print(f"SKIP: Could not load dataset — {exc}")
        sys.exit(0)

    documents: list[Document] = []
    test_questions: list[str] = []

    for row in list(ds)[:30]:
        ctx_list = (row.get("context") or {}).get("contexts") or []
        for ctx in ctx_list:
            if ctx and len(ctx) > 100:
                documents.append(Document(
                    page_content=ctx[:600],
                    metadata={"source": "pubmed"},
                ))
        if row.get("question"):
            test_questions.append(row["question"])
        if len(documents) >= 60:
            break

    if not documents:
        print("SKIP: No usable abstract segments loaded from dataset")
        sys.exit(0)

    test_question = test_questions[0] if test_questions else "What are the treatment outcomes?"

    print(f"  Loaded {len(documents)} real PubMed abstract segments")
    print(f"  Example: {documents[0].page_content[:100]}...")
    print(f"  Test question: {test_question[:80]}...")

    from ragfallback.diagnostics import ChunkQualityChecker

    checker = ChunkQualityChecker(min_chars=80)
    report = checker.check(documents)
    print(f"\nChunkQualityChecker: {report.n_chunks} segments  Violations: {len(report.violations)}")

    import _kb_common

    emb = _kb_common.get_embeddings()

    try:
        from langchain_community.vectorstores import FAISS
    except ImportError:
        print("SKIP: pip install ragfallback[faiss] for vector store")
        sys.exit(0)

    print("\nBuilding FAISS index...")
    vs = FAISS.from_documents(documents, emb)
    print(f"  Indexed {len(documents)} real abstract segments")

    print("\nRetrieval demo (no LLM required)...")
    questions = [test_question, "What are the side effects?", "What does the study conclude?"]
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    for q in questions[:2]:
        hits = retriever.invoke(q)
        best = hits[0].page_content[:90] if hits else "—"
        print(f"  Q: {q[:60]}...")
        print(f"     → {best}...")

    print("\n✅ Medical RAG demo complete (no paid API keys used).")
    print("   To add LLM synthesis: pass an LLM to AdaptiveRAGRetriever.")


if __name__ == "__main__":
    main()
