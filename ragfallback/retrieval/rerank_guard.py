"""
Placeholder for cross-encoder / reranking (semantic noise after top-k).

Production systems often add a second-stage reranker. Hook your model here later.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

from langchain_core.documents import Document


class ReRankerGuard:
    """
    Pass-through today; call ``rerank_fn(query, docs) -> List[Document]`` when set.

    Intended to sit after vector retrieval and before the LLM prompt.
    """

    def __init__(
        self,
        rerank_fn: Optional[Callable[[str, List[Document]], List[Document]]] = None,
        top_n: Optional[int] = None,
    ):
        self.rerank_fn = rerank_fn
        self.top_n = top_n

    def apply(self, query: str, documents: Sequence[Document]) -> List[Document]:
        docs = list(documents)
        if self.rerank_fn is not None:
            docs = self.rerank_fn(query, docs)
        if self.top_n is not None:
            docs = docs[: self.top_n]
        return docs
