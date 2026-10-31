"""Deterministic offline stand-ins for embeddings, LLMs, and retrievers used in unit tests."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

_WORD = re.compile(r"[a-z0-9]+")


class KeywordEmbeddings(Embeddings):
    """Bag-of-words hashed into ``dim`` buckets: texts sharing words get high cosine."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for w in _WORD.findall(text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        if not any(v):
            v[0] = 1e-3
        return v

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


class FakeChatResponse:
    def __init__(self, content: str, usage: dict[str, int] | None = None) -> None:
        self.content = content
        self.response_metadata = {"token_usage": usage} if usage else {}


class ScriptedLLM:
    """Returns queued responses in order (last one repeats); records prompts."""

    def __init__(self, *responses: str, model_name: str = "gpt-4o-mini") -> None:
        self.responses = list(responses) or ["ok"]
        self.model_name = model_name
        self.prompts: list[Any] = []

    def invoke(self, messages: Any, **_: Any) -> FakeChatResponse:
        self.prompts.append(messages)
        text = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return FakeChatResponse(text, {"prompt_tokens": 100, "completion_tokens": 20})


class ListRetriever:
    """Minimal duck-typed retriever returning a fixed list."""

    def __init__(self, docs: list[Document]) -> None:
        self.docs = docs

    def invoke(self, query: str, **_: Any) -> list[Document]:
        return list(self.docs)


class StoreOf:
    """Vector-store shim whose as_retriever() returns a fixed retriever."""

    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever

    def as_retriever(self, **_: Any) -> Any:
        return self.retriever
