"""Retrieval / generation evaluation helpers (extend with LLM-as-judge as needed)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

from ragfallback.tracking.metrics import MetricsCollector


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = list(retrieved_ids)[:k]
    hits = sum(1 for rid in top if rid in relevant_ids)
    return hits / len(relevant_ids)


def dcg_at_k(relevance_scores: Sequence[float], k: int) -> float:
    """Binary or graded relevance in rank order (best first)."""
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k]):
        gain = (2**rel - 1) if rel > 0 else 0.0
        dcg += gain / math.log2(i + 2)
    return dcg


def ndcg_at_k(relevance_scores: Sequence[float], ideal_scores: Sequence[float], k: int) -> float:
    dcg = dcg_at_k(relevance_scores, k)
    idcg = dcg_at_k(sorted(ideal_scores, reverse=True), k)
    if idcg <= 0:
        return 0.0
    return dcg / idcg


@dataclass
class RAGEvalSummary:
    retrieval: Dict[str, float] = field(default_factory=dict)
    generation: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {"retrieval": self.retrieval, "generation": self.generation, "notes": self.notes}


def _tokenize_words(text: str) -> Set[str]:
    return set(re.findall(r"\b\w+\b", (text or "").lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokenize_words(a), _tokenize_words(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _parse_llm_score_0_10(raw: str) -> float:
    nums = re.findall(r"\b(10|[0-9])\b", (raw or "").strip())
    if nums:
        return int(nums[0]) / 10.0
    return 0.5


def _invoke_llm_text(llm: Any, prompt: str) -> str:
    try:
        from langchain_core.messages import HumanMessage

        res = llm.invoke([HumanMessage(content=prompt)])
        if hasattr(res, "content"):
            return str(res.content).strip()
        return str(res).strip()
    except Exception:
        out = llm(prompt) if callable(llm) else ""
        return str(out).strip()


_FAITH_PROMPT = """Rate how well the answer is grounded in the context (0-10 only).
10 = claims are directly supported; 0 = unsupported or contradictory.
Reply with one integer 0-10.

Question: {question}
Answer: {answer}
Context: {context}

Score:"""

_REL_PROMPT = """Rate how well the answer addresses the question (0-10 only).
Reply with one integer 0-10.

Question: {question}
Answer: {answer}

Score:"""


@dataclass
class RAGScore:
    """Single-example RAG quality view (heuristic + optional LLM judge)."""

    question: str
    answer: str
    context_precision: float = 0.0
    faithfulness_score: float = 0.0
    answer_relevance: float = 0.0
    recall_at_k: Optional[float] = None
    num_contexts: int = 0
    used_llm_judge: bool = False
    warnings: List[str] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        parts = [
            (self.context_precision, 0.35),
            (self.faithfulness_score, 0.40),
            (self.answer_relevance, 0.25),
        ]
        if self.recall_at_k is not None:
            parts.append((self.recall_at_k, 0.30))
        wsum = sum(w for _, w in parts)
        return sum(v * w for v, w in parts) / wsum if wsum else 0.0

    @property
    def passed(self) -> bool:
        return self.overall_score >= 0.70

    def report(self) -> str:
        lines = [
            "=" * 56,
            " RAG evaluation",
            "=" * 56,
            f" Context precision  : {self.context_precision:.2%}",
            f" Faithfulness       : {self.faithfulness_score:.2%}",
            f" Answer relevance   : {self.answer_relevance:.2%}",
        ]
        if self.recall_at_k is not None:
            lines.append(f" Recall (gold hit)  : {self.recall_at_k:.2%}")
        lines += [
            f" Overall            : {self.overall_score:.2%}",
            f" Pass (>=70%)       : {self.passed}",
            f" LLM judge          : {self.used_llm_judge}",
        ]
        if self.warnings:
            lines.append(" Warnings:")
            for w in self.warnings:
                lines.append(f"   - {w}")
        lines.append("=" * 56)
        return "\n".join(lines)


@dataclass
class SimpleRAGReport:
    """
    Lightweight offline signals (not a substitute for LLM-as-judge or human eval).
    """

    context_precision: float
    answer_grounded_overlap: float
    overall: float
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"context_precision≈{self.context_precision:.0%} | "
            f"answer/context_overlap≈{self.answer_grounded_overlap:.0%} | "
            f"overall≈{self.overall:.0%}"
        )


class RAGEvaluator:
    """
    Labeled retrieval metrics (recall@k, nDCG) plus optional **evaluate()** with
    context precision, faithfulness, and answer relevance (heuristic or LLM-as-judge).
    """

    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        *,
        llm: Optional[Any] = None,
    ):
        self.metrics_collector = metrics_collector
        self._llm = llm
        self._eval_history: List[RAGEvalSummary] = []

    def evaluate_retrieval_case(
        self,
        retrieved_ids: Sequence[str],
        relevant_ids: Set[str],
        k_values: Sequence[int] = (1, 3, 5, 10),
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k in k_values:
            out[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
        return out

    def evaluate_ranked_relevance(
        self,
        relevance_per_rank: Sequence[float],
        k: int = 10,
    ) -> Dict[str, float]:
        """relevance_per_rank[i] = relevance of item at rank i (e.g. 1 or 0)."""
        ideal = sorted(relevance_per_rank, reverse=True)
        return {
            f"dcg@{k}": dcg_at_k(relevance_per_rank, k),
            f"ndcg@{k}": ndcg_at_k(relevance_per_rank, ideal, k),
        }

    def faithfulness_score(
        self,
        answer: str,
        context: str,
        judge: Optional[Any] = None,
    ) -> Optional[float]:
        """
        Placeholder for LLM-as-judge faithfulness in [0,1]. Pass a callable
        ``judge(answer, context) -> float`` when ready.
        """
        if judge is None:
            return None
        try:
            return float(judge(answer, context))
        except Exception:
            return None

    def context_precision(
        self,
        retrieved_relevant_flags: Sequence[bool],
    ) -> float:
        """Fraction of retrieved items that are relevant (precision over list)."""
        if not retrieved_relevant_flags:
            return 0.0
        return sum(1 for x in retrieved_relevant_flags if x) / len(
            retrieved_relevant_flags
        )

    def record_summary(self, summary: RAGEvalSummary) -> None:
        self._eval_history.append(summary)

    def get_history(self) -> List[RAGEvalSummary]:
        return list(self._eval_history)

    def evaluate_heuristic(
        self,
        answer: str,
        contexts: Sequence[Any],
        ground_truth: Optional[str] = None,
    ) -> SimpleRAGReport:
        """
        Heuristic scoring without an LLM: token overlap answer↔context and optional answer↔gold.
        Use for regression smoke tests; pair with :meth:`faithfulness_score` + judge for quality.
        """
        notes: List[str] = []
        ctx_texts = [
            (getattr(c, "page_content", str(c)) or "") for c in contexts
        ]
        blob = "\n".join(ctx_texts).lower()
        ans = (answer or "").lower()

        def _tok_overlap(a: str, b: str) -> float:
            ta = set(re.findall(r"[a-z0-9]{3,}", a))
            tb = set(re.findall(r"[a-z0-9]{3,}", b))
            if not ta or not tb:
                return 0.0
            return len(ta & tb) / len(ta)

        overlap = _tok_overlap(ans, blob)
        if not ctx_texts:
            notes.append("no contexts provided")
        cp = overlap if ctx_texts else 0.0

        gt_sim = 1.0
        if ground_truth:
            gt_sim = _tok_overlap(ans, ground_truth.lower())

        overall = 0.6 * cp + 0.4 * gt_sim if ground_truth else cp
        return SimpleRAGReport(
            context_precision=cp,
            answer_grounded_overlap=overlap,
            overall=min(1.0, overall),
            notes=notes,
        )

    def _context_precision_answer(self, answer: str, contexts: List[str]) -> float:
        if not contexts:
            return 0.0
        hit = sum(1 for ctx in contexts if _jaccard(answer, ctx) > 0.15)
        return hit / len(contexts)

    def _heuristic_faithfulness(self, answer: str, contexts: List[str]) -> float:
        if not contexts:
            return 0.0
        combined = " ".join(contexts)
        return min(_jaccard(answer, combined) * 2.0, 1.0)

    def _recall_ground_truth_in_contexts(
        self, ground_truth: str, contexts: List[str]
    ) -> float:
        if not contexts or not ground_truth.strip():
            return 0.0
        gt = _tokenize_words(ground_truth)
        if not gt:
            return 0.0
        for ctx in contexts:
            cw = _tokenize_words(ctx)
            if len(gt & cw) / len(gt) > 0.5:
                return 1.0
        return 0.0

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: Sequence[Any],
        ground_truth: Optional[str] = None,
    ) -> RAGScore:
        """
        Score one RAG turn. ``contexts`` may be strings or objects with ``page_content``.
        With ``llm`` passed to the constructor, uses short 0–10 judge prompts; else heuristics.
        """
        warnings: List[str] = []
        ctx_list = [
            (c if isinstance(c, str) else (getattr(c, "page_content", str(c)) or ""))
            for c in contexts
        ]
        used_llm = self._llm is not None

        cp = self._context_precision_answer(answer, ctx_list)

        if self._llm:
            comb = "\n\n".join(ctx_list[:5])[:3000]
            faith = _parse_llm_score_0_10(
                _invoke_llm_text(
                    self._llm,
                    _FAITH_PROMPT.format(
                        question=question, answer=answer, context=comb
                    ),
                )
            )
            rel = _parse_llm_score_0_10(
                _invoke_llm_text(
                    self._llm, _REL_PROMPT.format(question=question, answer=answer)
                )
            )
        else:
            faith = self._heuristic_faithfulness(answer, ctx_list)
            rel = _jaccard(question, answer)
            warnings.append("No judge LLM — using heuristic faithfulness and relevance.")

        recall = None
        if ground_truth:
            recall = self._recall_ground_truth_in_contexts(ground_truth, ctx_list)

        if not ctx_list:
            warnings.append("No contexts — scores are unreliable.")
        if not (answer or "").strip():
            warnings.append("Empty answer.")

        return RAGScore(
            question=question,
            answer=answer,
            context_precision=cp,
            faithfulness_score=faith,
            answer_relevance=rel,
            recall_at_k=recall,
            num_contexts=len(ctx_list),
            used_llm_judge=used_llm,
            warnings=warnings,
        )

    def evaluate_batch(self, samples: Sequence[Dict[str, Any]]) -> List[RAGScore]:
        """Each dict: question, answer, contexts, optional ground_truth."""
        out: List[RAGScore] = []
        for s in samples:
            out.append(
                self.evaluate(
                    question=s["question"],
                    answer=s.get("answer", ""),
                    contexts=s.get("contexts", []),
                    ground_truth=s.get("ground_truth"),
                )
            )
        return out

    def batch_summary(self, scores: Sequence[RAGScore]) -> str:
        if not scores:
            return "No scores."
        n = len(scores)
        avg_o = sum(s.overall_score for s in scores) / n
        pr = sum(1 for s in scores if s.passed) / n
        return (
            f"batch n={n} pass_rate={pr:.1%} avg_overall={avg_o:.2%} "
            f"avg_faith={sum(s.faithfulness_score for s in scores)/n:.2%}"
        )
