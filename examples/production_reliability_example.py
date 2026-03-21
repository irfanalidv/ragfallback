"""
USE CASE: UC-1–UC-7 — Full reliability pipeline (single script)
===============================================================
Real problem : Silent failures at chunk, embed, index, retrieve, context, and answer layers.
Goal         : Chunk check → EmbeddingGuard → Chroma → stale/content manifest →
               retrieval smoke → context budget → RAGEvaluator (Mistral for judge).
Module       : ragfallback.diagnostics, ragfallback.evaluation, ragfallback.utils
Vector DB    : ChromaDB (local)
Install      : pip install ragfallback[mistral,chroma,sentence-transformers]
Env vars     : MISTRAL_API_KEY in .env for LLM judge step
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from langchain_core.documents import Document

from ragfallback.diagnostics import (
    ChunkQualityChecker,
    ContextWindowGuard,
    EmbeddingGuard,
    RetrievalHealthCheck,
    StaleIndexDetector,
)
from ragfallback.evaluation import RAGEvaluator
from ragfallback.utils import (
    create_chroma_vector_store,
    create_llm_from_env,
    create_open_source_embeddings,
    load_env,
)


def main() -> None:
    load_env()

    raw = [
        "Python is a high-level language known for readability and a large ecosystem.",
        "Short.",  # too short
        "Lists are mutable; dictionaries map keys to values " * 3,
    ]

    print("=== 1. ChunkQualityChecker ===")
    cq = ChunkQualityChecker(min_chars=80, min_words=8)
    rep = cq.check(raw)
    print(rep.summary())
    fixed = cq.auto_fix(raw)
    print(f"auto_fix: {len(raw)} -> {len(fixed)} strings")

    docs = [
        Document(page_content=t, metadata={"source": f"chunk_{i}"})
        for i, t in enumerate(fixed)
    ]

    print("\n=== 2. EmbeddingGuard (live model) ===")
    emb = create_open_source_embeddings()
    g = EmbeddingGuard()
    vr = g.validate(emb, expected_dim=None)
    print(vr.message)

    print("\n=== 3. Vector store + content manifest ===")
    persist = Path(tempfile.mkdtemp(prefix="ragfallback_prod_")) / "chroma"
    vs = create_chroma_vector_store(
        documents=docs,
        embeddings=emb,
        persist_directory=str(persist),
        collection_name="reliability_demo",
    )

    det = StaleIndexDetector()
    man = Path(tempfile.mkdtemp(prefix="ragfallback_man_")) / "content.json"
    det.record_document_contents(docs, manifest_path=man)
    drift = det.check_document_contents(docs, manifest_path=man)
    print(drift.summary())

    print("\n=== 4. RetrievalHealthCheck.quick_check ===")
    health = RetrievalHealthCheck(k=4)
    qc = health.quick_check(vs, docs, sample_size=min(3, len(docs)), seed=42)
    print(qc.summary())

    print("\n=== 5. ContextWindowGuard ===")
    cwg = ContextWindowGuard(max_context_tokens=2000)
    retrieved = vs.similarity_search("Python programming", k=3)
    packed, ctx_rep = cwg.select(
        "Python programming", retrieved, emb, lost_in_middle=True
    )
    print(ctx_rep.summary(), f"selected={len(packed)}")

    print("\n=== 6. RAGEvaluator ===")
    llm = create_llm_from_env(temperature=0, load_dotenv=False)
    ev = RAGEvaluator(llm=llm)
    answer = "Python is a high-level programming language."
    score = ev.evaluate(
        question="What is Python?",
        answer=answer,
        contexts=[d.page_content for d in retrieved],
        ground_truth="Python is a high-level language",
    )
    print(score.report())


if __name__ == "__main__":
    main()
