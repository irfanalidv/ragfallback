"""
Smart threshold on vector scores + optional BM25 fallback (hybrid reliability).

Addresses "semantic noise": top-k can inject irrelevant chunks. Filter by similarity /
distance or relative-to-top, then fall back to BM25 for keywords/acronyms when the vector
pass returns nothing.

Install BM25: ``pip install rank_bm25`` (optional; required only if you use bm25_retriever).
"""

from __future__ import annotations

import logging
from typing import Any, List, Literal, Optional, Sequence, Tuple

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

logger = logging.getLogger(__name__)

ScoreMode = Literal["distance", "similarity", "relative_similarity"]


def _annotate(docs: List[Document], source: str) -> List[Document]:
    out = []
    for d in docs:
        md = dict(d.metadata or {})
        md["ragfallback_retrieval_source"] = source
        out.append(Document(page_content=d.page_content, metadata=md))
    return out


class SmartThresholdHybridRetriever(BaseRetriever):
    """
    1. ``similarity_search_with_score`` on the vector store (if available) with ``fetch_k``.
    2. Filter by threshold (see ``score_mode``).
    3. If no docs remain and ``bm25_retriever`` is set, run BM25.
    4. If still empty, return ``[]`` (safe empty — caller / LLM prompt should treat as no context).

    If the store has no ``similarity_search_with_score``, falls back to plain
    ``similarity_search`` with ``k`` (no threshold), then BM25 if empty.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectorstore: Any
    bm25_retriever: Optional[Any] = None
    fetch_k: int = 20
    final_k: int = 5
    score_mode: ScoreMode = "distance"
    # For score_mode == "distance" (lower is better, e.g. L2): keep if score <= max_distance
    max_distance: Optional[float] = None
    # For score_mode == "similarity" (higher is better, e.g. cosine sim): keep if score >= min_similarity
    min_similarity: Optional[float] = 0.7
    # For score_mode == "relative_similarity": keep docs with score >= best * relative_floor
    relative_floor: float = 0.85

    def _dense_scores_are_weak(self, docs: List[Document]) -> bool:
        """Return ``True`` when dense results are insufficient and BM25 fallback should run.

        This is the decision gate used inside :meth:`_get_relevant_documents` before
        engaging the BM25 retriever.  Override in a subclass to implement custom
        quality criteria (e.g. require at least ``k`` results, or check a minimum
        score stored in metadata).

        Args:
            docs: Filtered documents returned by the dense retriever after score
                thresholding.

        Returns:
            ``True`` if the dense results are empty (no document survived the
            threshold), ``False`` otherwise.
        """
        return len(docs) == 0

    def _filter_scored(
        self, pairs: Sequence[Tuple[Document, float]]
    ) -> List[Document]:
        if not pairs:
            return []
        scores = [s for _, s in pairs]

        if self.score_mode == "relative_similarity":
            best = max(scores)
            cutoff = best * self.relative_floor
            kept = [(d, s) for d, s in pairs if s >= cutoff]
        elif self.score_mode == "similarity":
            thr = self.min_similarity if self.min_similarity is not None else 0.0
            kept = [(d, s) for d, s in pairs if s >= thr]
        else:  # distance
            if self.max_distance is None:
                kept = list(pairs)
            else:
                kept = [(d, s) for d, s in pairs if s <= self.max_distance]

        kept.sort(key=lambda x: x[1], reverse=(self.score_mode != "distance"))
        docs = [d for d, _ in kept[: self.final_k]]
        return docs

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        vs = self.vectorstore
        docs: List[Document] = []

        if hasattr(vs, "similarity_search_with_score"):
            try:
                pairs = vs.similarity_search_with_score(query, k=self.fetch_k)
            except Exception as e:
                logger.warning("similarity_search_with_score failed: %s", e)
                pairs = []
            docs = self._filter_scored(pairs)
        else:
            try:
                if hasattr(vs, "similarity_search"):
                    docs = vs.similarity_search(query, k=self.final_k)
                else:
                    ret = vs.as_retriever(search_kwargs={"k": self.final_k})
                    docs = ret.invoke(query)
            except Exception as e:
                logger.warning("vector fallback retrieve failed: %s", e)
                docs = []

        if not self._dense_scores_are_weak(docs):
            return _annotate(docs, "vector_threshold")

        if self.bm25_retriever is not None:
            try:
                invoke = getattr(self.bm25_retriever, "invoke", None)
                if invoke:
                    bm_docs = invoke(query)
                else:
                    bm_docs = self.bm25_retriever.get_relevant_documents(query)
                if isinstance(bm_docs, list):
                    return _annotate(bm_docs[: self.final_k], "bm25_fallback")
            except Exception as e:
                logger.warning("BM25 fallback failed: %s", e)

        return []

    @classmethod
    def from_documents(
        cls,
        documents: Sequence[Document],
        vectorstore: Any,
        *,
        fetch_k: int = 20,
        final_k: int = 5,
        score_mode: ScoreMode = "distance",
        max_distance: Optional[float] = None,
        min_similarity: Optional[float] = 0.7,
        relative_floor: float = 0.85,
    ) -> SmartThresholdHybridRetriever:
        """
        Build with an optional in-memory BM25 retriever from the same corpus (requires rank_bm25).
        """
        bm25 = None
        try:
            from langchain_community.retrievers import BM25Retriever  # lazy optional dep

            bm25 = BM25Retriever.from_documents(list(documents))
            bm25.k = final_k
        except ImportError as exc:
            raise ImportError(
                "BM25 hybrid requires: pip install ragfallback[hybrid]\n"
                "(installs rank_bm25 and langchain-community)"
            ) from exc

        return cls(
            vectorstore=vectorstore,
            bm25_retriever=bm25,
            fetch_k=fetch_k,
            final_k=final_k,
            score_mode=score_mode,
            max_distance=max_distance,
            min_similarity=min_similarity,
            relative_floor=relative_floor,
        )
