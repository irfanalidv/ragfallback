"""
USE CASE: UC-2 — Embedding guard before index write
=====================================================
Real problem : Switching embedding models changes vector dimension; stores fail cryptically.
Goal         : Validate live embedder dimension matches the expected index width.
Module       : ragfallback.diagnostics.EmbeddingGuard
Vector DB    : N/A (validation only)
Install      : pip install ragfallback[sentence-transformers]
Env vars     : NONE required
"""

from __future__ import annotations

import _kb_common
from ragfallback.diagnostics import EmbeddingGuard


def main() -> None:
    emb = _kb_common.get_embeddings()
    guard = EmbeddingGuard(expected_dim=384)
    report = guard.validate(emb)
    print(report.message)
    report.raise_if_failed()


if __name__ == "__main__":
    main()
