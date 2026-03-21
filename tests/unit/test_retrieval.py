"""Tests for hybrid retrieval and production helpers."""

from langchain_core.documents import Document

from ragfallback.diagnostics.context_stitcher import OverlappingContextStitcher
from ragfallback.diagnostics.schema_sanitizer import sanitize_documents
from ragfallback.retrieval.failover import FailoverRetriever
from ragfallback.retrieval.smart_hybrid import SmartThresholdHybridRetriever


class MockVectorStoreDistance:
    def similarity_search_with_score(self, query: str, k: int = 10):
        return [
            (Document(page_content="close match"), 0.15),
            (Document(page_content="weak match"), 0.95),
        ]


def test_smart_threshold_filters_by_distance():
    r = SmartThresholdHybridRetriever(
        vectorstore=MockVectorStoreDistance(),
        bm25_retriever=None,
        fetch_k=10,
        final_k=5,
        score_mode="distance",
        max_distance=0.5,
    )
    out = r.invoke("query")
    assert len(out) == 1
    assert out[0].page_content == "close match"
    assert out[0].metadata.get("ragfallback_retrieval_source") == "vector_threshold"


def test_smart_threshold_safe_empty():
    class AllFar:
        def similarity_search_with_score(self, query: str, k: int = 10):
            return [(Document(page_content="x"), 99.0)]

    r = SmartThresholdHybridRetriever(
        vectorstore=AllFar(),
        bm25_retriever=None,
        score_mode="distance",
        max_distance=0.5,
        fetch_k=5,
        final_k=3,
    )
    assert r.invoke("q") == []


def test_failover_primary_ok():
    class OkR:
        def invoke(self, q):
            return [Document(page_content="ok")]

    class BadR:
        def invoke(self, q):
            raise RuntimeError("never")

    f = FailoverRetriever(primary=OkR(), secondary=BadR(), log_errors=False)
    docs = f.invoke("x")
    assert len(docs) == 1
    assert docs[0].metadata.get("ragfallback_retrieval_source") == "primary"


def test_failover_secondary_on_error():
    class Bad:
        def invoke(self, q):
            raise ConnectionError("down")

    class Ok:
        def invoke(self, q):
            return [Document(page_content="backup")]

    f = FailoverRetriever(primary=Bad(), secondary=Ok(), log_errors=False)
    docs = f.invoke("x")
    assert docs[0].page_content == "backup"
    assert docs[0].metadata.get("ragfallback_retrieval_source") == "secondary_failover"


def test_sanitize_nested_metadata():
    docs = [
        Document(
            page_content="a",
            metadata={"ok": 1, "nested": ["unhashable", {"k": "v"}]},
        )
    ]
    clean = sanitize_documents(docs)
    assert isinstance(clean[0].metadata["nested"], str)


def test_stitch_adjacent():
    d0 = Document(page_content="A", metadata={"source": "f", "chunk_index": 0})
    d1 = Document(page_content="B", metadata={"source": "f", "chunk_index": 1})
    s = OverlappingContextStitcher()
    merged = s.stitch([d1, d0])
    assert len(merged) == 1
    assert "A" in merged[0].page_content and "B" in merged[0].page_content
