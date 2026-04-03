"""Optional Ragas integration with heuristic fallback via RAGEvaluator."""

from __future__ import annotations

import asyncio
import functools
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from ragfallback.evaluation.rag_evaluator import RAGEvaluator, RAGScore

logger = logging.getLogger(__name__)


@dataclass
class RagasReport:
    """Aggregated scores from Ragas or from RAGEvaluator fallback."""

    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
    n_samples: int
    fallback_mode: bool
    timestamp: datetime
    raw: Dict[str, Any]


class RagasHook:
    """Run Ragas metrics when installed; otherwise average RAGEvaluator scores."""

    def __init__(
        self,
        llm: Any,
        embeddings: Any,
        threshold: float = 0.05,
    ) -> None:
        """Store LLM/embeddings, probe Ragas import, and build fallback evaluator."""
        self.llm = llm
        self.embeddings = embeddings
        self.threshold = threshold
        self._ragas_available = False
        try:
            import ragas  # noqa: F401

            self._ragas_available = True
        except ImportError:
            logger.warning(
                "ragas not installed — falling back to RAGEvaluator heuristics. "
                "Install with: pip install ragfallback[mlops]"
            )
        self._fallback_evaluator = RAGEvaluator(llm=llm)

    def _to_ragas_dataset(self, samples: List[Dict[str, Any]]) -> Any:
        """Build Ragas EvaluationDataset from dict samples."""
        try:
            from ragas import EvaluationDataset
            try:
                from ragas.dataset_schema import SingleTurnSample
            except ImportError:
                from ragas import SingleTurnSample
        except ImportError as exc:
            raise RuntimeError("ragas dataset types unavailable") from exc

        sts: List[Any] = []
        for s in samples:
            ctx = s.get("contexts")
            if ctx is None:
                ctx = s.get("retrieved_contexts") or []
            if isinstance(ctx, str):
                ctx = [ctx]
            ref = s.get("ground_truth")
            if ref is None:
                ref = s.get("reference")
            sts.append(
                SingleTurnSample(
                    user_input=s.get("question", ""),
                    retrieved_contexts=list(ctx),
                    response=s.get("answer", ""),
                    reference=ref or "",
                )
            )
        return EvaluationDataset(samples=sts)

    def _fallback_evaluate(self, samples: List[Dict[str, Any]]) -> RagasReport:
        """Average RAGScore fields into a Ragas-shaped report."""
        batch: List[Dict[str, Any]] = []
        for s in samples:
            ctx = s.get("contexts")
            if ctx is None:
                ctx = s.get("retrieved_contexts") or []
            batch.append(
                {
                    "question": s.get("question", ""),
                    "answer": s.get("answer", ""),
                    "contexts": ctx,
                    "ground_truth": s.get("ground_truth") or s.get("reference"),
                }
            )
        scores: List[RAGScore] = self._fallback_evaluator.evaluate_batch(batch)
        n = len(scores) or 1
        faith = sum(s.faithfulness_score for s in scores) / n
        ans_rel = sum(s.answer_relevance for s in scores) / n
        ctx_prec = sum(s.context_precision for s in scores) / n
        ctx_rec_vals = [
            (s.recall_at_k if s.recall_at_k is not None else 0.0) for s in scores
        ]
        ctx_rec = sum(ctx_rec_vals) / n
        return RagasReport(
            faithfulness=faith,
            answer_relevance=ans_rel,
            context_precision=ctx_prec,
            context_recall=ctx_rec,
            n_samples=len(scores),
            fallback_mode=True,
            timestamp=datetime.utcnow(),
            raw={},
        )

    def evaluate_sync(self, samples: List[Dict[str, Any]]) -> RagasReport:
        """Evaluate samples with Ragas or heuristic fallback."""
        if not samples:
            return RagasReport(
                faithfulness=0.0,
                answer_relevance=0.0,
                context_precision=0.0,
                context_recall=0.0,
                n_samples=0,
                fallback_mode=not self._ragas_available,
                timestamp=datetime.utcnow(),
                raw={},
            )
        if not self._ragas_available:
            return self._fallback_evaluate(samples)
        try:
            from ragas import evaluate
            try:
                from ragas.metrics import (
                    answer_relevancy,
                    context_precision,
                    context_recall,
                    faithfulness,
                )
                answer_relevance_metric = answer_relevancy
            except ImportError:
                from ragas.metrics import (
                    answer_relevance as answer_relevance_metric,
                    context_precision,
                    context_recall,
                    faithfulness,
                )

            dataset = self._to_ragas_dataset(samples)
            result = evaluate(
                dataset=dataset,
                metrics=[
                    faithfulness,
                    answer_relevance_metric,
                    context_precision,
                    context_recall,
                ],
                llm=self.llm,
                embeddings=self.embeddings,
            )
            raw_dict: Dict[str, Any] = {}
            faith = ans_rel = ctx_p = ctx_r = 0.0
            try:
                pdf = result.to_pandas()
                raw_dict = pdf.to_dict()

                def _mean(*names: str) -> float:
                    for n in names:
                        if n in pdf.columns:
                            return float(pdf[n].astype(float).mean())
                    return 0.0

                faith = _mean("faithfulness")
                ans_rel = _mean("answer_relevancy", "answer_relevance")
                ctx_p = _mean("context_precision")
                ctx_r = _mean("context_recall")
            except Exception:
                if isinstance(result, dict):
                    raw_dict = dict(result)
                else:
                    raw_dict = {"repr": repr(result)}

            return RagasReport(
                faithfulness=float(faith),
                answer_relevance=float(ans_rel),
                context_precision=float(ctx_p),
                context_recall=float(ctx_r),
                n_samples=len(samples),
                fallback_mode=False,
                timestamp=datetime.utcnow(),
                raw=raw_dict,
            )
        except Exception as exc:
            logger.warning("ragas evaluation failed, using fallback: %s", exc)
            return self._fallback_evaluate(samples)

    async def evaluate_async(self, samples: List[Dict[str, Any]]) -> RagasReport:
        """Run :meth:`evaluate_sync` in a thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, functools.partial(self.evaluate_sync, samples)
        )
