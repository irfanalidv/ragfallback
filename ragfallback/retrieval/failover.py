"""Try a primary retriever; on too-few results or failure, use a fallback."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, model_validator

logger = logging.getLogger(__name__)


class FailoverRetriever(BaseRetriever):
    """Return results from *primary*; switch to *fallback* automatically when the
    primary returns fewer than *min_results* documents **or** raises an exception.

    This is a drop-in replacement for any LangChain retriever and provides a
    circuit-breaker style safety net for hosted vector-store outages or edge-case
    queries that produce empty results from the primary index.

    Both ``fallback`` (preferred, per spec) and ``secondary`` (legacy alias) are
    accepted for backward compatibility.

    Example::

        fb = FailoverRetriever(
            primary=pinecone_retriever,
            fallback=local_faiss_retriever,
            min_results=1,
        )
        docs = fb.invoke("billing refund policy")
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    primary: Any
    # ``fallback`` is the canonical name; ``secondary`` is kept for backward compat.
    fallback: Optional[Any] = None
    secondary: Optional[Any] = None
    min_results: int = 1
    log_errors: bool = True

    @model_validator(mode="after")
    def _resolve_fallback_retriever(self) -> "FailoverRetriever":
        """Ensure at least one of ``fallback`` or ``secondary`` is set, and sync them."""
        if self.fallback is not None and self.secondary is None:
            object.__setattr__(self, "secondary", self.fallback)
        elif self.secondary is not None and self.fallback is None:
            object.__setattr__(self, "fallback", self.secondary)
        if self.secondary is None and self.fallback is None:
            raise ValueError(
                "FailoverRetriever requires a fallback retriever. "
                "Pass fallback=<retriever> (or the legacy secondary=<retriever>)."
            )
        return self

    @property
    def _effective_fallback(self) -> Any:
        """Return the active fallback retriever (``fallback`` takes precedence)."""
        return self.fallback if self.fallback is not None else self.secondary

    def _invoke_retriever(self, retriever: Any, query: str) -> List[Document]:
        """Call *retriever* with *query*, preferring ``invoke`` over the legacy method."""
        invoke = getattr(retriever, "invoke", None)
        if invoke is not None:
            out = invoke(query)
            return out if isinstance(out, list) else []
        return retriever.get_relevant_documents(query)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Retrieve documents, falling over to the fallback when necessary.

        Args:
            query: The search query string.
            run_manager: LangChain callback manager (passed by the base class).

        Returns:
            Documents from the primary retriever when it returns at least
            ``min_results`` results, otherwise documents from the fallback retriever.
        """
        try:
            docs = self._invoke_retriever(self.primary, query)
            if len(docs) >= self.min_results:
                logger.debug(
                    "FailoverRetriever: primary returned %d doc(s) — using primary.",
                    len(docs),
                )
                return [
                    Document(
                        page_content=d.page_content,
                        metadata={**(d.metadata or {}), "ragfallback_retrieval_source": "primary"},
                    )
                    for d in docs
                ]
            logger.debug(
                "FailoverRetriever: primary returned %d doc(s) (< min_results=%d) "
                "— switching to fallback.",
                len(docs),
                self.min_results,
            )
        except Exception as exc:
            if self.log_errors:
                logger.warning("FailoverRetriever: primary raised — switching to fallback: %s", exc)

        try:
            docs = self._invoke_retriever(self._effective_fallback, query)
            logger.debug(
                "FailoverRetriever: fallback returned %d doc(s).", len(docs)
            )
            return [
                Document(
                    page_content=d.page_content,
                    metadata={
                        **(d.metadata or {}),
                        "ragfallback_retrieval_source": "secondary_failover",
                    },
                )
                for d in docs
            ]
        except Exception as exc:
            logger.error("FailoverRetriever: fallback also failed: %s", exc)
            return []
