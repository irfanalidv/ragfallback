"""
A CI regression gate for a RAG pipeline, end to end, in a few seconds.

What it shows
    ``GoldenRunner`` drives an ``AdaptiveRAGRetriever`` over a small golden set,
    ``RagasHook`` scores it (heuristic ``RAGEvaluator`` fallback when ragas is not
    installed), and ``BaselineRegistry`` stores the result as the baseline. A second
    run against a *degraded* index - one help-centre article accidentally dropped
    from ingestion - is compared with ``compare_or_fail`` and raises
    ``RegressionError``, which is what fails the build in CI.

What it needs
    Nothing beyond ``pip install ragfallback``. No network, no API key: bundled
    ``examples/sample_kb`` articles, a hashing embedder, and the rule-based stub LLM
    from ``examples/_offline.py``. For real scores install ``ragfallback[mlops]`` and
    pass a real LLM + embeddings to ``RagasHook``.

Run
    python examples/offline_eval_gate.py
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _offline import ExtractiveStubLLM, HashingEmbeddings, load_paragraphs
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from ragfallback import AdaptiveRAGRetriever
from ragfallback.mlops import (
    BaselineRegistry,
    GoldenReport,
    GoldenRunner,
    RagasHook,
    RegressionError,
)
from ragfallback.strategies import QueryVariationsStrategy

GOLDEN = [
    {
        "query": "How long is the refund window?",
        "ground_truth": "30 days",
        "relevant_doc_ids": ["billing_refunds.md"],
    },
    {
        "query": "When do API tokens expire?",
        "ground_truth": "90 days",
        "relevant_doc_ids": ["api_authentication.md"],
    },
    {
        "query": "What status code is returned when the rate limit is exceeded?",
        "ground_truth": "429",
        "relevant_doc_ids": ["rate_limits.md", "api_authentication.md"],
    },
    {
        "query": "How many dimensions do the embedding vectors have?",
        "ground_truth": "384",
        "relevant_doc_ids": ["embedding_index_policy.md"],
    },
]


def build_pipeline(docs: list[Document]) -> AdaptiveRAGRetriever:
    embeddings = HashingEmbeddings()
    store = InMemoryVectorStore(embedding=embeddings)
    store.add_documents(docs)
    return AdaptiveRAGRetriever(
        vector_store=store,
        llm=ExtractiveStubLLM(),  # type: ignore[arg-type]
        embedding_model=embeddings,
        fallback_strategies=[QueryVariationsStrategy(num_variations=1)],
        max_attempts=2,
        min_confidence=0.7,
        enable_logging=False,
    )


def show(label: str, report: GoldenReport) -> None:
    print(
        f"   {label:<9} recall@3={report.recall_at_3:.2f} "
        f"faithfulness={report.ragas.faithfulness:.2f} "
        f"context_recall={report.ragas.context_recall:.2f} "
        f"fallback_rate={report.fallback_rate:.0%}"
    )


def main() -> int:
    logging.getLogger("ragfallback").setLevel(logging.ERROR)  # hide "ragas not installed"
    docs = load_paragraphs()
    for d in docs:  # GoldenRunner reads ids from metadata id / doc_id / source
        d.metadata["id"] = d.metadata["source"]

    hook = RagasHook(llm=None, embeddings=None)
    baselines = BaselineRegistry(str(Path(tempfile.mkdtemp()) / "baselines.json"))

    print("1) Healthy pipeline -> record baseline")
    healthy = GoldenRunner(build_pipeline(docs), hook, GOLDEN).run()
    show("healthy", healthy)
    baselines.compare_or_fail(healthy, dataset="support-kb")  # no baseline yet: passes
    baselines.update(healthy, dataset="support-kb")

    print("\n2) Ingestion bug drops billing_refunds.md -> gate")
    degraded_docs = [d for d in docs if d.metadata["source"] != "billing_refunds.md"]
    degraded = GoldenRunner(build_pipeline(degraded_docs), hook, GOLDEN).run()
    show("degraded", degraded)
    try:
        baselines.compare_or_fail(degraded, dataset="support-kb", latency_threshold=10.0)
    except RegressionError as err:
        print(f"\n   RegressionError (CI would exit 1 here):\n{err}")
        return 0
    print("   no regression detected")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
