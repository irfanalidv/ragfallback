"""
USE CASE: UC-1 + UC-6 — Qdrant local server demo
=================================================
Real problem : Most examples use in-process vector DBs; production often uses a
               network vector service. One script should prove ragfallback works
               against HTTP Qdrant.
Goal         : Index with qdrant-client, run ``RetrievalHealthCheck.quick_check``,
               pack context with ``ContextWindowGuard``, heuristic ``RAGEvaluator``.
Module       : ragfallback.diagnostics / evaluation (Qdrant via thin shim)
Vector DB    : Qdrant (Docker, localhost:6333)
Install      : pip install ragfallback[qdrant,sentence-transformers]
Docker       : docker run -p 6333:6333 qdrant/qdrant
Env vars     : NONE required
"""

from __future__ import annotations

import sys
from typing import Any, List, Optional

import _kb_common
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from ragfallback.diagnostics import ContextWindowGuard, RetrievalHealthCheck
from ragfallback.evaluation import RAGEvaluator


class QdrantVectorShim:
    """Minimal vector surface for ragfallback checks (qdrant-client only, no langchain-qdrant)."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embeddings: Embeddings,
    ) -> None:
        self._client = client
        self._collection = collection_name
        self._emb = embeddings

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        qv = self._emb.embed_query(query)
        hits = self._client.search(
            collection_name=self._collection,
            query_vector=list(qv),
            limit=k,
        )
        out: List[Document] = []
        for h in hits:
            payload = h.payload or {}
            text = str(payload.get("page_content", ""))
            meta = {a: b for a, b in payload.items() if a != "page_content"}
            out.append(Document(page_content=text, metadata=meta))
        return out

    def as_retriever(self, search_kwargs: Optional[dict[str, Any]] = None) -> Any:
        sk = search_kwargs or {}
        k = int(sk.get("k", 4))
        parent = self

        class _R:
            def invoke(self, query: str) -> List[Document]:
                return parent.similarity_search(query, k=k)

            def get_relevant_documents(self, query: str) -> List[Document]:
                return parent.similarity_search(query, k=k)

        return _R()


def _ensure_qdrant() -> QdrantClient:
    client = QdrantClient(host="localhost", port=6333)
    try:
        client.get_collections()
    except Exception:
        raise ConnectionError(
            "Qdrant is not running.\n"
            "Start it with: docker run -p 6333:6333 qdrant/qdrant\n"
            "Then re-run this script."
        ) from None
    return client


def main() -> int:
    client = _ensure_qdrant()
    docs = _kb_common.load_sample_kb()
    emb = _kb_common.get_embeddings()
    dim = len(emb.embed_query("dimension probe"))
    collection = "ragfallback_qdrant_demo"

    if collection in [c.name for c in client.get_collections().collections]:
        client.delete_collection(collection_name=collection)

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    texts = [d.page_content for d in docs]
    vectors = emb.embed_documents(texts)
    points: List[PointStruct] = []
    for i, (doc, vec) in enumerate(zip(docs, vectors)):
        payload = {"page_content": doc.page_content, **(doc.metadata or {})}
        points.append(
            PointStruct(id=i + 1, vector=list(vec), payload=payload),
        )
    client.upsert(collection_name=collection, points=points)

    shim = QdrantVectorShim(client, collection, emb)
    health = RetrievalHealthCheck(k=4)
    report = health.quick_check(shim, docs, sample_size=min(10, len(docs)), seed=7)
    print(health.__class__.__name__, report.summary())
    if report.avg_latency_ms is not None:
        print(f"avg_latency_ms={report.avg_latency_ms:.1f}")

    q = "refund policy billing"
    retrieved = shim.similarity_search(q, k=15)
    guard = ContextWindowGuard.from_model_name("gpt-3.5-turbo")
    packed, ctx_rep = guard.select(q, retrieved, emb)
    print(ctx_rep.summary(), f"packed={len(packed)}")
    assert ctx_rep.used_tokens <= ctx_rep.budget_tokens

    ev = RAGEvaluator(llm=None)
    ans = "Refunds are processed within several business days per policy."
    score = ev.evaluate(
        question=q,
        answer=ans,
        contexts=[d.page_content for d in packed[:5]],
    )
    print(score.report())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConnectionError as e:
        print(e, file=sys.stderr)
        raise SystemExit(1)
