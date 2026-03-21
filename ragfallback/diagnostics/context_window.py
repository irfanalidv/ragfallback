"""Rank and trim retrieved chunks to fit an approximate context budget."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Approximate **document** token budgets (leave headroom for system + answer).
CONTEXT_WINDOW_TOKEN_PRESETS = {
    "gpt-3.5-turbo": 3000,
    "gpt-4": 7000,
    "gpt-4-turbo": 100000,
    "gpt-4o": 100000,
    "gpt-4o-mini": 100000,
    "claude-3-opus": 150000,
    "claude-3-sonnet": 150000,
    "claude-3-haiku": 150000,
    "llama-3": 7500,
    "mistral": 12000,
    "default": 6000,
}


def _estimate_tokens(
    text: str,
    chars_per_token: float = 4.0,
    encoding_name: Optional[str] = None,
) -> int:
    if not text:
        return 0
    if encoding_name:
        try:
            import tiktoken

            enc = tiktoken.encoding_for_model(encoding_name)
            return max(1, len(enc.encode(text)))
        except Exception:
            pass
    return max(1, int(len(text) / chars_per_token))


def _cosine(a: List[float], b: List[float]) -> float:
    if np is None:
        # pure Python fallback
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
    va, vb = np.array(a), np.array(b)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


@dataclass
class ContextWindowReport:
    budget_tokens: int
    used_tokens: int
    n_selected: int
    n_dropped: int
    dropped_reasons: List[str] = field(default_factory=list)

    def summary(self) -> str:
        pct = (self.used_tokens / self.budget_tokens * 100) if self.budget_tokens else 0
        return (
            f"context_window kept {self.n_selected}/{(self.n_selected + self.n_dropped)} chunks "
            f"~{self.used_tokens}/{self.budget_tokens} est. tokens ({pct:.0f}% of budget)"
        )


class ContextWindowGuard:
    """
    Select a subset of documents for the LLM prompt: score by embedding similarity to the
    query, greedily pack by estimated token count until ``max_context_tokens``.
    """

    def __init__(
        self,
        max_context_tokens: int = 6000,
        chars_per_token: float = 4.0,
        separator_tokens: int = 4,
        tiktoken_model: Optional[str] = None,
    ):
        self.max_context_tokens = max_context_tokens
        self.chars_per_token = chars_per_token
        self.separator_tokens = separator_tokens
        self.tiktoken_model = tiktoken_model

    @classmethod
    def from_model_name(
        cls,
        model: str,
        *,
        reserve_ratio: float = 0.12,
        chars_per_token: float = 4.0,
        separator_tokens: int = 4,
    ) -> "ContextWindowGuard":
        """
        Pick a conservative document budget from :data:`CONTEXT_WINDOW_TOKEN_PRESETS`.
        ``reserve_ratio`` reserves capacity for system prompt + user question + completion.
        """
        key = (model or "default").strip().lower()
        raw = CONTEXT_WINDOW_TOKEN_PRESETS.get(key)
        if raw is None:
            for prefix, budget in CONTEXT_WINDOW_TOKEN_PRESETS.items():
                if prefix != "default" and key.startswith(prefix):
                    raw = budget
                    break
        if raw is None:
            raw = CONTEXT_WINDOW_TOKEN_PRESETS["default"]
        budget = max(512, int(raw * (1.0 - max(0.0, min(0.5, reserve_ratio)))))
        return cls(
            max_context_tokens=budget,
            chars_per_token=chars_per_token,
            separator_tokens=separator_tokens,
        )

    def select(
        self,
        query: str,
        documents: Sequence[Document],
        embeddings: Embeddings,
        *,
        lost_in_middle: bool = False,
    ) -> Tuple[List[Document], ContextWindowReport]:
        if not documents:
            return [], ContextWindowReport(
                budget_tokens=self.max_context_tokens,
                used_tokens=0,
                n_selected=0,
                n_dropped=0,
                dropped_reasons=["no documents"],
            )

        q_vec = embeddings.embed_query(query)
        scored: List[Tuple[float, Document]] = []
        for doc in documents:
            text = doc.page_content or ""
            d_vec = embeddings.embed_query(text[:2000])
            scored.append((_cosine(q_vec, d_vec), doc))
        scored.sort(key=lambda x: -x[0])

        selected: List[Document] = []
        used = 0
        reasons: List[str] = []
        for score, doc in scored:
            t = _estimate_tokens(
                doc.page_content or "",
                self.chars_per_token,
                self.tiktoken_model,
            )
            cost = t + (self.separator_tokens if selected else 0)
            if used + cost > self.max_context_tokens:
                reasons.append(f"skipped doc (est_tokens={t}, score={score:.3f})")
                continue
            selected.append(doc)
            used += cost

        if lost_in_middle and len(selected) >= 3:
            # Place 2nd-most-relevant at the end (weak "lost in the middle" mitigation).
            first, second, rest = selected[0], selected[1], selected[2:]
            selected = [first] + rest + [second]

        return selected, ContextWindowReport(
            budget_tokens=self.max_context_tokens,
            used_tokens=used,
            n_selected=len(selected),
            n_dropped=len(documents) - len(selected),
            dropped_reasons=reasons[:20],
        )
