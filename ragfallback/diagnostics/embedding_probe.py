"""Heuristic probe: query–document similarity spread (generic vs domain mismatch hint)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

from langchain_core.embeddings import Embeddings


def _cosine_list(a: List[float], b: List[float]) -> float:
    if np is None:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
    va, vb = np.array(a), np.array(b)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


@dataclass
class EmbeddingQualityReport:
    ok: bool
    mean_top1_similarity: float
    spread: float
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"embedding_probe ok={self.ok} mean_top1_sim={self.mean_top1_similarity:.3f} "
            f"spread={self.spread:.3f}"
        )


class EmbeddingQualityProbe:
    """
    Embed the query and a small set of reference snippets. If the best cosine similarity
    stays below ``min_mean_top1``, warn that the embedder may be a poor domain match
    (heuristic only — tune thresholds per corpus).
    """

    def __init__(
        self,
        min_mean_top1: float = 0.25,
        max_spread: float = 0.15,
    ):
        self.min_mean_top1 = min_mean_top1
        self.max_spread = max_spread

    def run(
        self,
        embeddings: Embeddings,
        query: str,
        reference_snippets: Sequence[str],
    ) -> EmbeddingQualityReport:
        warnings: List[str] = []
        if not reference_snippets:
            return EmbeddingQualityReport(
                ok=False,
                mean_top1_similarity=0.0,
                spread=0.0,
                warnings=["no reference_snippets provided"],
            )

        qv = embeddings.embed_query(query)
        sims: List[float] = []
        for snip in reference_snippets:
            if not snip.strip():
                continue
            dv = embeddings.embed_query(snip[:2000])
            sims.append(_cosine_list(qv, dv))

        if not sims:
            return EmbeddingQualityReport(
                ok=False,
                mean_top1_similarity=0.0,
                spread=0.0,
                warnings=["all reference snippets empty"],
            )

        top1 = max(sims)
        mean_sim = sum(sims) / len(sims)
        spread = max(sims) - min(sims)

        if mean_sim < self.min_mean_top1:
            warnings.append(
                f"low mean similarity to references ({mean_sim:.3f} < {self.min_mean_top1}); "
                "consider a domain-specific embedding model or fine-tuning."
            )
        if spread > self.max_spread:
            warnings.append(
                f"high spread across references ({spread:.3f}); corpus may be heterogeneous "
                "or queries misaligned with indexed text."
            )

        ok = not warnings
        return EmbeddingQualityReport(
            ok=ok,
            mean_top1_similarity=top1,
            spread=spread,
            warnings=warnings,
        )
