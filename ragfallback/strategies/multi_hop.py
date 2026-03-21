"""Multi-hop fallback strategy — decompose complex questions into retrieval chains."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage

from ragfallback.strategies.base import FallbackStrategy

__all__ = [
    "MultiHopFallbackStrategy",
    "MultiHopResult",
    "HopResult",
]

_logger = logging.getLogger(__name__)

_DECOMPOSE_PROMPT = (
    'You are a question decomposer. Break this complex question into at most {max_hops} '
    'simpler sub-questions that can each be answered by searching a document store '
    'independently. Each sub-question should target a single concrete fact.\n\n'
    'Complex question: "{question}"\n\n'
    'Return ONLY a JSON array of strings. No explanations, no markdown.\n'
    'Example: ["What company acquired Acme Corp?", "What was that company\'s revenue?"]'
)

_HOP_ANSWER_PROMPT = (
    "Answer this specific question using ONLY the context provided. Be concise.\n\n"
    "Sub-question: {question}\n\n"
    "Context:\n{context}\n\n"
    "If the answer is not in the context, reply with exactly: NOT_FOUND\n"
    "Answer:"
)

_SYNTHESIS_PROMPT = (
    "You are synthesising evidence from multiple retrieval steps to answer a "
    "multi-part question.\n\n"
    'Original question: "{original_question}"\n\n'
    "Evidence gathered:\n{evidence}\n\n"
    "Provide a concise, complete answer based on the evidence above. "
    "If any part of the answer is missing, say so clearly."
)


@dataclass
class HopResult:
    """Evidence gathered during one retrieval step of a multi-hop chain."""

    hop_number: int
    sub_question: str
    retrieved_chunks: List[str]
    partial_answer: str
    confidence: float


@dataclass
class MultiHopResult:
    """Aggregated output from a complete multi-hop retrieval run."""

    final_answer: str
    hops: List[HopResult]
    total_hops: int
    success: bool

    def summary(self) -> str:
        """Return a one-line human-readable summary.

        Returns:
            Status, hop count, and a truncated preview of the final answer.
        """
        status = "SUCCESS" if self.success else "FAILED"
        preview = self.final_answer[:80]
        if len(self.final_answer) > 80:
            preview += "…"
        return f"[{status}] hops={self.total_hops} answer='{preview}'"


class MultiHopFallbackStrategy(FallbackStrategy):
    """Decomposes a complex question into sub-questions, retrieves evidence per hop,
    and synthesises a final answer.

    Use this when single-shot retrieval consistently fails on questions that chain
    facts across multiple documents — e.g. *"What was the revenue of the company
    that acquired Acme Corp?"* requires knowing *who* acquired Acme before you can
    retrieve their revenue.

    The strategy satisfies the :class:`FallbackStrategy` contract (``generate_queries``)
    so it can be plugged directly into :class:`~ragfallback.AdaptiveRAGRetriever`.
    For full orchestration with per-hop retrieval and synthesis, call :meth:`run`
    directly.

    Example::

        strategy = MultiHopFallbackStrategy(max_hops=3, top_k=4)

        # As a FallbackStrategy (inside AdaptiveRAGRetriever):
        sub_qs = strategy.generate_queries(question, context={}, attempt=2, llm=llm)

        # As a standalone pipeline:
        result = strategy.run(question, retriever=my_retriever, llm=my_llm)
        print(result.summary())
    """

    def __init__(self, max_hops: int = 3, top_k: int = 4) -> None:
        """
        Args:
            max_hops: Maximum number of sub-questions (retrieval hops). Must be >= 1.
                Use :class:`QueryVariationsStrategy` for single-hop query rewriting.
            top_k: Number of chunks to retrieve per sub-question.

        Raises:
            ValueError: If *max_hops* is less than 1.
        """
        if max_hops < 1:
            raise ValueError(
                f"max_hops must be >= 1, got {max_hops}. "
                "Use QueryVariationsStrategy for single-hop fallback."
            )
        self.max_hops = max_hops
        self.top_k = top_k

    # ------------------------------------------------------------------
    # FallbackStrategy contract
    # ------------------------------------------------------------------

    def generate_queries(
        self,
        original_query: str,
        context: Dict[str, Any],
        attempt: int,
        llm: BaseLanguageModel,
    ) -> List[str]:
        """Decompose *original_query* into sub-questions for independent retrieval.

        Called by :class:`~ragfallback.AdaptiveRAGRetriever` on each fallback attempt.
        Returns the original query unchanged on the first attempt so the caller can
        try a direct hit before escalating to multi-hop decomposition.

        Args:
            original_query: The user's full multi-part question.
            context: Optional context dict forwarded by the retriever (unused here).
            attempt: Current attempt number (1-indexed). Returns ``[original_query]``
                unchanged when ``attempt == 1``.
            llm: Language model used for decomposition.

        Returns:
            List of sub-question strings (at most ``max_hops``), or
            ``[original_query]`` when decomposition fails.
        """
        if attempt == 1:
            return [original_query]

        sub_questions = self._decompose(original_query, llm)
        if not sub_questions:
            _logger.warning(
                "MultiHopFallbackStrategy: decomposition returned no sub-questions; "
                "falling back to original query."
            )
            return [original_query]

        _logger.info(
            "MultiHopFallbackStrategy: decomposed into %d sub-questions.",
            len(sub_questions),
        )
        return sub_questions

    # ------------------------------------------------------------------
    # Direct orchestration API
    # ------------------------------------------------------------------

    def run(
        self,
        question: str,
        retriever: Any,
        llm: BaseLanguageModel,
    ) -> MultiHopResult:
        """Execute multi-hop retrieval and synthesis end-to-end.

        Args:
            question: The original complex question.
            retriever: Any object with ``.invoke(query)`` or
                ``.get_relevant_documents(query)`` returning a list of
                ``langchain_core.documents.Document``-compatible objects.
            llm: Language model for decomposition, hop answers, and synthesis.

        Returns:
            :class:`MultiHopResult` containing per-hop evidence and the
            synthesised final answer.
        """
        sub_questions = self._decompose(question, llm)
        if not sub_questions:
            return MultiHopResult(
                final_answer="Could not decompose the question into sub-questions.",
                hops=[],
                total_hops=0,
                success=False,
            )

        hops: List[HopResult] = []
        for i, sub_q in enumerate(sub_questions, start=1):
            _logger.info("Hop %d/%d: %s", i, len(sub_questions), sub_q)
            chunks = self._retrieve(retriever, sub_q)
            partial, conf = self._answer_hop(sub_q, chunks, llm)
            hops.append(
                HopResult(
                    hop_number=i,
                    sub_question=sub_q,
                    retrieved_chunks=chunks,
                    partial_answer=partial,
                    confidence=conf,
                )
            )

        final = self._synthesise(question, hops, llm)
        success = (
            any(h.confidence >= 0.5 for h in hops)
            and bool(final)
            and final != "NOT_FOUND"
        )
        return MultiHopResult(
            final_answer=final,
            hops=hops,
            total_hops=len(hops),
            success=success,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decompose(self, question: str, llm: BaseLanguageModel) -> List[str]:
        """Ask the LLM to break *question* into up to ``max_hops`` sub-questions."""
        prompt = _DECOMPOSE_PROMPT.format(question=question, max_hops=self.max_hops)
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            text = response.content if hasattr(response, "content") else str(response)
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(q) for q in parsed[: self.max_hops] if q]
            _logger.warning(
                "_decompose: expected JSON list, got %s; attempting text extraction.",
                type(parsed).__name__,
            )
        except json.JSONDecodeError:
            text = (
                response.content  # type: ignore[possibly-undefined]
                if hasattr(response, "content")  # type: ignore[possibly-undefined]
                else str(response)  # type: ignore[possibly-undefined]
            )
            sub_qs = _extract_list_from_text(text)
            if sub_qs:
                return sub_qs[: self.max_hops]
            _logger.warning(
                "_decompose: JSON parse failed; raw response: %.200s", text
            )
        except Exception as exc:
            _logger.error("_decompose: LLM call failed: %s", exc)
        return []

    def _retrieve(self, retriever: Any, query: str) -> List[str]:
        """Return up to ``top_k`` ``page_content`` strings from *retriever*."""
        try:
            invoke = getattr(retriever, "invoke", None)
            docs = invoke(query) if invoke is not None else retriever.get_relevant_documents(query)
            if not isinstance(docs, list):
                return []
            return [
                str(getattr(d, "page_content", d) or "")
                for d in docs[: self.top_k]
                if d is not None
            ]
        except Exception as exc:
            _logger.error("_retrieve: failed for query %r: %s", query[:80], exc)
            return []

    def _answer_hop(
        self, sub_question: str, chunks: List[str], llm: BaseLanguageModel
    ) -> tuple[str, float]:
        """Generate a partial answer for one hop and estimate confidence.

        Returns:
            A ``(partial_answer, confidence)`` tuple. Confidence is ``0.0``
            when no chunks were retrieved or the LLM returns ``NOT_FOUND``.
        """
        if not chunks:
            return "NOT_FOUND", 0.0

        context_str = "\n\n".join(
            f"[{i + 1}] {c}" for i, c in enumerate(chunks) if c.strip()
        )
        prompt = _HOP_ANSWER_PROMPT.format(question=sub_question, context=context_str)
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            text = (
                response.content if hasattr(response, "content") else str(response)
            ).strip()
            if text.upper() == "NOT_FOUND" or not text:
                return "NOT_FOUND", 0.0
            confidence = 0.8 if len(text) > 20 else 0.5
            return text, confidence
        except Exception as exc:
            _logger.error("_answer_hop: LLM call failed: %s", exc)
            return "NOT_FOUND", 0.0

    def _synthesise(
        self, original_question: str, hops: List[HopResult], llm: BaseLanguageModel
    ) -> str:
        """Combine all hop evidence into a single final answer."""
        evidence_parts = []
        for hop in hops:
            answer = hop.partial_answer if hop.partial_answer != "NOT_FOUND" else "(not found)"
            evidence_parts.append(
                f"Sub-question {hop.hop_number}: {hop.sub_question}\nAnswer: {answer}"
            )
        evidence_str = "\n\n".join(evidence_parts)
        prompt = _SYNTHESIS_PROMPT.format(
            original_question=original_question, evidence=evidence_str
        )
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            return (
                response.content if hasattr(response, "content") else str(response)
            ).strip()
        except Exception as exc:
            _logger.error("_synthesise: LLM call failed: %s", exc)
            found = [h.partial_answer for h in hops if h.partial_answer != "NOT_FOUND"]
            return (
                " | ".join(found)
                if found
                else "Could not synthesise answer from gathered evidence."
            )


def _extract_list_from_text(text: str) -> List[str]:
    """Best-effort extraction of a list from non-JSON LLM output."""
    import re

    m = re.search(r"\[(.*?)\]", text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(f"[{m.group(1)}]")
            if isinstance(parsed, list):
                return [str(q) for q in parsed if q]
        except json.JSONDecodeError:
            pass
    lines = [
        ln.strip().lstrip("-•*123456789. ").strip('"').strip("'")
        for ln in text.splitlines()
    ]
    return [ln for ln in lines if len(ln) > 10]
