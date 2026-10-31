"""Small compatibility helpers shared across subpackages (private)."""

from __future__ import annotations

from typing import Any


def invoke_retriever(retriever: Any, query: str) -> list[Any]:
    """Run ``retriever`` on ``query`` and return a list of documents.

    Prefers the Runnable ``invoke()`` API (LangChain >= 0.1). Falls back to the legacy
    ``get_relevant_documents()`` for duck-typed retrievers that predate it; that method
    no longer exists on ``langchain_core.retrievers.BaseRetriever`` in LangChain 1.x.
    """
    invoke = getattr(retriever, "invoke", None)
    if invoke is not None:
        return list(invoke(query) or [])
    legacy = getattr(retriever, "get_relevant_documents", None)
    if legacy is None:
        raise TypeError(
            f"{type(retriever).__name__} has neither invoke() nor get_relevant_documents()"
        )
    return list(legacy(query) or [])
