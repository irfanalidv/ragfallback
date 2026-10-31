"""Shims so LangChain components that expect a ``VectorStore`` can use a custom retriever."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ragfallback._compat import invoke_retriever


class RetrieverAsVectorStore:
    """
    Minimal ``as_retriever()`` surface for
    :class:`~ragfallback.core.adaptive_retriever.AdaptiveRAGRetriever`.

    Example::

        hybrid = SmartThresholdHybridRetriever(vectorstore=vs, bm25_retriever=bm25, ...)
        shim = RetrieverAsVectorStore(hybrid)
        AdaptiveRAGRetriever(vector_store=shim, llm=llm, embedding_model=emb)
    """

    def __init__(self, retriever: BaseRetriever):
        self._retriever = retriever

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        return self._retriever

    # Optional: some code paths call similarity_search directly — not supported here.
    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        return invoke_retriever(self._retriever, query)[:k]
