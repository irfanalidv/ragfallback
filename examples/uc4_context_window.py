"""
USE CASE: UC-4 — Context window budget
========================================
Real problem : Too many retrieved chunks exceed the LLM window; truncation hides the best docs.
Goal         : Rank/pack chunks with ``ContextWindowGuard`` to stay under a token budget.
Module       : ragfallback.diagnostics.ContextWindowGuard
Vector DB    : ChromaDB (local) for retrieval; guard is model-agnostic
Install      : pip install ragfallback[chroma,sentence-transformers]
Env vars     : NONE required
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import _kb_common
from ragfallback.diagnostics import ContextWindowGuard


def main() -> None:
    docs = _kb_common.load_sample_kb()
    persist = Path(tempfile.mkdtemp(prefix="ragfallback_uc4_")) / "chroma"
    vs = _kb_common.build_chroma_store(docs, persist_directory=persist, collection_name="uc4")
    emb = _kb_common.get_embeddings()
    q = "refund policy authentication"
    retrieved = vs.similarity_search(q, k=12)
    guard = ContextWindowGuard.from_model_name("gpt-3.5-turbo")
    packed, rep = guard.select(q, retrieved, emb, lost_in_middle=True)
    print(rep.summary(), f"selected={len(packed)}")
    assert rep.used_tokens <= rep.budget_tokens, "context overflow"


if __name__ == "__main__":
    main()
