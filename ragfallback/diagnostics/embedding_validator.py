"""Pre-flight embedding dimension vs vector store metadata (extends :class:`EmbeddingGuard`)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from langchain_core.embeddings import Embeddings

from ragfallback.diagnostics.embedding_guard import EmbeddingGuard, EmbeddingGuardReport

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingValidatorReport:
    ok: bool
    guard: EmbeddingGuardReport
    store_reported_dim: Optional[int]
    message: str


def infer_vectorstore_embedding_dim(vectorstore: Any) -> Optional[int]:
    """
    Best-effort read of collection/index embedding size (Chroma metadata, etc.).
    Many stores do not expose this — returns None then rely on :class:`EmbeddingGuard` only.
    """
    try:
        col = getattr(vectorstore, "_collection", None)
        if col is not None:
            meta = getattr(col, "metadata", None)
            if callable(meta):
                meta = meta()
            if isinstance(meta, dict):
                for key in (
                    "embedding_dimensionality",
                    "dimension",
                    "dims",
                    "embedding_dimension",
                ):
                    if key in meta and meta[key] is not None:
                        return int(meta[key])
    except (TypeError, ValueError, AttributeError):
        pass

    try:
        idx = getattr(vectorstore, "index", None)
        if idx is not None and hasattr(idx, "d"):
            return int(idx.d)
    except (TypeError, ValueError, AttributeError):
        pass

    return None


class EmbeddingValidator:
    """
    Compare live embedding model width to optional store-reported dimension, then
    :class:`EmbeddingGuard`. On mismatch, logs a clear **model / index version** message.
    Re-initializing the correct embedder is left to application code (factory swap).
    """

    def __init__(self, embedding_guard: Optional[EmbeddingGuard] = None):
        self._guard = embedding_guard or EmbeddingGuard()

    def validate(
        self,
        embeddings: Embeddings,
        vectorstore: Any,
        sample_text: str = "__ragfallback_embedding_dim_probe__",
    ) -> EmbeddingValidatorReport:
        store_dim = infer_vectorstore_embedding_dim(vectorstore)
        guard_rep = self._guard.validate(
            embeddings, expected_dim=store_dim, sample_text=sample_text
        )

        if store_dim is None:
            msg = (
                f"{guard_rep.message} "
                "(Vector store did not report embedding dimension; "
                "set expected_dim explicitly when you know the index build model.)"
            )
            ok = guard_rep.ok
        elif guard_rep.ok:
            msg = guard_rep.message
            ok = True
        else:
            msg = (
                f"Model / index mismatch: {guard_rep.message} "
                "Re-index with the current embedding model or point the app at the "
                "embedder used when the collection was built."
            )
            ok = False
            logger.error("EmbeddingValidator: %s", msg)

        return EmbeddingValidatorReport(
            ok=ok,
            guard=guard_rep,
            store_reported_dim=store_dim,
            message=msg,
        )
