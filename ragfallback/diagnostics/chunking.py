"""Chunk quality checks before indexing (size, overlap hints, boundary heuristics)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

_SENTENCE_END = re.compile(r"[.!?…][\"']?\s*$")

if TYPE_CHECKING:
    from langchain_core.documents import Document
else:
    Document = object  # type: ignore


def _chunk_texts(chunks: Sequence[Union[str, Document]]) -> List[str]:
    out: List[str] = []
    for c in chunks:
        if isinstance(c, str):
            out.append(c)
        elif hasattr(c, "page_content"):
            out.append(getattr(c, "page_content", "") or "")
        else:
            out.append(str(c))
    return out


@dataclass
class ChunkQualityReport:
    """Result of :class:`ChunkQualityChecker` analysis."""

    ok: bool
    n_chunks: int
    violations: List[str] = field(default_factory=list)
    char_lengths: List[int] = field(default_factory=list)
    avg_length: float = 0.0
    min_length: int = 0
    max_length: int = 0
    estimated_overlap_ratio: Optional[float] = None
    avg_words: float = 0.0
    mid_sentence_ratio: Optional[float] = None

    @property
    def has_issues(self) -> bool:
        """True when any chunking rule failed (same as ``not ok``)."""
        return not self.ok

    def summary(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        parts = [
            f"[{status}] chunks={self.n_chunks}",
            f"len min/avg/max={self.min_length}/{self.avg_length:.0f}/{self.max_length}",
        ]
        if self.estimated_overlap_ratio is not None:
            parts.append(f"overlap≈{self.estimated_overlap_ratio:.2%}")
        if self.violations:
            parts.append("violations: " + "; ".join(self.violations[:5]))
        return " | ".join(parts)


class ChunkQualityChecker:
    """
    Flag chunking issues that hurt retrieval (size, harsh cuts, weak overlap).

    Overlap ratio is estimated only for *ordered* adjacent pairs: shared prefix/suffix
    length vs shorter chunk length (heuristic, not tokenizer-accurate).
    """

    def __init__(
        self,
        min_chars: int = 100,
        max_chars: int = 8000,
        min_words: int = 10,
        target_overlap_ratio: float = 0.15,
        overlap_tolerance: float = 0.08,
        require_sentence_end: bool = False,
        mid_sentence_warn_ratio: float = 0.55,
    ):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.min_words = min_words
        self.target_overlap_ratio = target_overlap_ratio
        self.overlap_tolerance = overlap_tolerance
        self.require_sentence_end = require_sentence_end
        self.mid_sentence_warn_ratio = mid_sentence_warn_ratio

    def check(self, chunks: Sequence[Union[str, Document]]) -> ChunkQualityReport:
        texts = _chunk_texts(chunks)
        n = len(texts)
        lengths = [len(t) for t in texts]
        violations: List[str] = []

        if n == 0:
            return ChunkQualityReport(
                ok=False,
                n_chunks=0,
                violations=["no chunks provided"],
                char_lengths=[],
                avg_length=0.0,
                min_length=0,
                max_length=0,
                mid_sentence_ratio=None,
            )

        word_counts = [len((t or "").split()) for t in texts]
        mid_sentence = 0
        for i, L in enumerate(lengths):
            w = word_counts[i]
            if L < self.min_chars or w < self.min_words:
                violations.append(
                    f"chunk[{i}] too short ({L} chars, {w} words; "
                    f"min {self.min_chars} chars / {self.min_words} words)"
                )
            if L > self.max_chars:
                violations.append(f"chunk[{i}] too long ({L} > {self.max_chars})")
            stripped = texts[i].strip()
            # Only count longer spans as "mid-sentence" noise (titles/labels stay exempt).
            if len(stripped) >= 80 and not _SENTENCE_END.search(stripped):
                mid_sentence += 1
            if self.require_sentence_end and stripped:
                if stripped[-1] not in ".?!…\"":
                    violations.append(f"chunk[{i}] may end mid-sentence")
        ms_ratio = mid_sentence / n if n else 0.0
        if ms_ratio >= self.mid_sentence_warn_ratio:
            violations.append(
                f"high fraction of chunks end mid-sentence ({ms_ratio:.0%} ≥ {self.mid_sentence_warn_ratio:.0%})"
            )

        overlap_est: Optional[float] = None
        if n >= 2:
            ratios: List[float] = []
            for i in range(n - 1):
                a, b = texts[i], texts[i + 1]
                if not a or not b:
                    continue
                # Standard sliding-window overlap: suffix of a equals prefix of b
                max_l = min(len(a), len(b))
                overlap_chars = 0
                for L in range(max_l, 0, -1):
                    if a[-L:] == b[:L]:
                        overlap_chars = L
                        break
                shorter = min(len(a), len(b))
                if shorter > 0:
                    ratios.append(overlap_chars / shorter)
            if ratios:
                overlap_est = sum(ratios) / len(ratios)
                low = self.target_overlap_ratio - self.overlap_tolerance
                if overlap_est < low:
                    violations.append(
                        f"low estimated neighbor overlap ({overlap_est:.2%} < {low:.2%})"
                    )

        avg = sum(lengths) / n if n else 0.0
        avg_w = sum(word_counts) / n if n else 0.0
        ok = len(violations) == 0
        return ChunkQualityReport(
            ok=ok,
            n_chunks=n,
            violations=violations,
            char_lengths=lengths,
            avg_length=avg,
            min_length=min(lengths),
            max_length=max(lengths),
            estimated_overlap_ratio=overlap_est,
            avg_words=avg_w,
            mid_sentence_ratio=ms_ratio,
        )

    def suggest_fixes(self, report: ChunkQualityReport) -> List[str]:
        """Actionable hints from a report (does not mutate chunks)."""
        tips: List[str] = []
        if not report.violations:
            return tips
        vtext = " ".join(report.violations).lower()
        if "too short" in vtext:
            tips.append("Increase target chunk size or merge adjacent splits.")
        if "too long" in vtext:
            tips.append("Split large sections with overlap; cap max chunk length.")
        if "overlap" in vtext:
            tips.append("Raise chunk_overlap so neighbors share boundary text.")
        if "mid-sentence" in vtext or "sentence" in vtext:
            tips.append("Use a splitter that respects sentence boundaries (e.g. RecursiveCharacterTextSplitter).")
        return tips

    def auto_fix(
        self,
        chunks: Sequence[Union[str, Document]],
        target_chars: int = 500,
        overlap_chars: int = 80,
    ) -> List[str]:
        """
        Best-effort repair: merge undersized neighbors, split oversized segments.
        Returns plain strings — review before embedding.
        """
        texts = [t.strip() for t in _chunk_texts(chunks) if t.strip()]
        merged = _merge_small_chunks(texts, max(self.min_chars, 50))
        out: List[str] = []
        for t in merged:
            if len(t) > self.max_chars:
                out.extend(_split_large_text(t, target_chars, overlap_chars))
            else:
                out.append(t)
        return [s for s in out if s.strip()]


def _merge_small_chunks(texts: List[str], min_chars: int) -> List[str]:
    if not texts:
        return []
    result: List[str] = []
    buf = texts[0]
    for t in texts[1:]:
        if len(buf) < min_chars:
            buf = (buf + " " + t).strip()
        else:
            result.append(buf)
            buf = t
    result.append(buf)
    return result


def _split_large_text(text: str, target: int, overlap: int) -> List[str]:
    if len(text) <= target:
        return [text]
    parts: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + target)
        parts.append(text[start:end])
        if end >= n:
            break
        start = max(0, end - overlap)
    return parts
