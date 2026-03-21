"""Validate embedding model output before writing to a vector store."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

from langchain_core.embeddings import Embeddings

from ragfallback.exceptions import EmbeddingDimensionError


def _vector_is_sane(vec: List[float], min_norm: float = 1e-10) -> bool:
    """Return False if the vector is empty, contains NaN/Inf, or has near-zero norm."""
    if not vec:
        return False
    for x in vec:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return False
    norm = math.sqrt(sum(x * x for x in vec))
    return norm >= min_norm


@dataclass
class EmbeddingGuardReport:
    ok: bool
    observed_dim: int
    expected_dim: Optional[int]
    message: str

    def raise_if_failed(self) -> None:
        """Raise :class:`~ragfallback.exceptions.EmbeddingDimensionError` if validation failed."""
        if not self.ok:
            raise EmbeddingDimensionError(self.message)


class EmbeddingGuard:
    """
    Probe ``embed_query`` / ``embed_documents`` dimensionality and compare to an expected
    collection dimension (e.g. Chroma collection metadata). Surfaces clear errors before ingest.
    """

    def __init__(self, expected_dim: Optional[int] = None):
        self.expected_dim = expected_dim

    def probe_dimension(
        self,
        embeddings: Embeddings,
        sample_text: str = "__ragfallback_embedding_dim_probe__",
    ) -> int:
        vec = embeddings.embed_query(sample_text)
        if not vec:
            raise ValueError("embed_query returned an empty vector")
        if not _vector_is_sane(list(vec)):
            raise ValueError("embed_query returned a zero or non-finite vector")
        return len(vec)

    def validate(
        self,
        embeddings: Embeddings,
        expected_dim: Optional[int] = None,
        sample_text: str = "__ragfallback_embedding_dim_probe__",
    ) -> EmbeddingGuardReport:
        exp = expected_dim if expected_dim is not None else self.expected_dim
        dim = self.probe_dimension(embeddings, sample_text=sample_text)
        if exp is None:
            return EmbeddingGuardReport(
                ok=True,
                observed_dim=dim,
                expected_dim=None,
                message=f"Embedding dimension observed: {dim} (no expected_dim set).",
            )
        if dim != exp:
            msg = (
                f"Embedding dimension mismatch: model produces {dim} dims but "
                f"collection/index expects {exp}. Use the same embedder as at index "
                f"build time, or rebuild the collection with the new model."
            )
            return EmbeddingGuardReport(
                ok=False, observed_dim=dim, expected_dim=exp, message=msg
            )
        return EmbeddingGuardReport(
            ok=True,
            observed_dim=dim,
            expected_dim=exp,
            message=f"Embedding dimension {dim} matches expected {exp}.",
        )

    def validate_document_batch(
        self,
        embeddings: Embeddings,
        texts: Sequence[str],
        expected_dim: Optional[int] = None,
    ) -> EmbeddingGuardReport:
        """Ensure ``embed_documents`` returns consistent row width (first batch only)."""
        exp = expected_dim if expected_dim is not None else self.expected_dim
        if not texts:
            return EmbeddingGuardReport(
                ok=False,
                observed_dim=0,
                expected_dim=exp,
                message="No texts provided for embed_documents check.",
            )
        rows = embeddings.embed_documents(list(texts)[: min(8, len(texts))])
        if not rows:
            return EmbeddingGuardReport(
                ok=False,
                observed_dim=0,
                expected_dim=exp,
                message="embed_documents returned no vectors.",
            )
        for i, row in enumerate(rows):
            if not _vector_is_sane(list(row)):
                return EmbeddingGuardReport(
                    ok=False,
                    observed_dim=len(row) if row else 0,
                    expected_dim=exp,
                    message=f"embed_documents row[{i}] is empty, zero-norm, or non-finite.",
                )
        dims = {len(r) for r in rows}
        if len(dims) != 1:
            return EmbeddingGuardReport(
                ok=False,
                observed_dim=0,
                expected_dim=exp,
                message=f"Inconsistent embed_documents widths: {dims}",
            )
        dim = dims.pop()
        if exp is not None and dim != exp:
            msg = (
                f"embed_documents width {dim} != expected collection dimension {exp}. "
                "Rebuild the index or align the embedding model."
            )
            return EmbeddingGuardReport(
                ok=False, observed_dim=dim, expected_dim=exp, message=msg
            )
        return EmbeddingGuardReport(
            ok=True,
            observed_dim=dim,
            expected_dim=exp,
            message=f"embed_documents consistent width {dim}.",
        )

    @staticmethod
    def validate_raw_vectors(
        rows: Sequence[Sequence[float]],
        expected_dim: Optional[int] = None,
    ) -> EmbeddingGuardReport:
        """
        Validate pre-computed embedding rows (e.g. from a custom pipeline) before upsert.
        """
        if not rows:
            return EmbeddingGuardReport(
                ok=False,
                observed_dim=0,
                expected_dim=expected_dim,
                message="No embedding rows provided.",
            )
        dims = {len(list(r)) for r in rows}
        if len(dims) != 1:
            return EmbeddingGuardReport(
                ok=False,
                observed_dim=0,
                expected_dim=expected_dim,
                message=f"Inconsistent row widths: {dims}",
            )
        dim = dims.pop()
        for i, row in enumerate(rows[: min(16, len(rows))]):
            if not _vector_is_sane(list(row)):
                return EmbeddingGuardReport(
                    ok=False,
                    observed_dim=dim,
                    expected_dim=expected_dim,
                    message=f"Row {i} is zero-norm or non-finite.",
                )
        if expected_dim is not None and dim != expected_dim:
            return EmbeddingGuardReport(
                ok=False,
                observed_dim=dim,
                expected_dim=expected_dim,
                message=(
                    f"Vector width {dim} does not match expected dimension {expected_dim}."
                ),
            )
        return EmbeddingGuardReport(
            ok=True,
            observed_dim=dim,
            expected_dim=expected_dim,
            message=f"Validated {len(rows)} rows of width {dim}.",
        )
