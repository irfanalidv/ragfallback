"""Small compatibility helpers shared across subpackages (private)."""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import SecretStr


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


def content_hash(text: str) -> str:
    """Process-stable short hash of ``text`` (16 hex chars of SHA-256).

    Use this instead of the builtin ``hash()``, which is salted per interpreter run
    (PYTHONHASHSEED) and therefore unusable for ids that are persisted or compared
    across processes.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def secret(value: str | None) -> SecretStr | None:
    """Wrap an optional API key for LangChain clients, which type keys as ``SecretStr``."""
    return SecretStr(value) if value else None
