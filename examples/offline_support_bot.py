"""
Customer-support RAG that recovers from a vocabulary mismatch.

What it shows
    A customer asks "How long do I have to get my money back?". The help-centre
    articles never say "money back" - they say "refund window". A single dense
    lookup misses, the answer comes back "Not found", and ``AdaptiveRAGRetriever``
    rejects it on confidence, asks ``QueryVariationsStrategy`` for rewrites, retries,
    and returns the grounded answer. ``CostTracker`` and ``MetricsCollector`` record
    what the retries cost.

What it needs
    Nothing beyond ``pip install ragfallback``. No API key, no model download: it uses
    the bundled ``examples/sample_kb`` articles, a hashing embedder, and a rule-based
    stub LLM from ``examples/_offline.py``. Swap in a real chat model and embedder for
    production use.

Run
    python examples/offline_support_bot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _offline import ExtractiveStubLLM, HashingEmbeddings, load_paragraphs
from langchain_core.vectorstores import InMemoryVectorStore

from ragfallback import AdaptiveRAGRetriever, CostTracker, MetricsCollector
from ragfallback.strategies import QueryVariationsStrategy


def main() -> int:
    docs = load_paragraphs()
    embeddings = HashingEmbeddings()
    store = InMemoryVectorStore(embedding=embeddings)
    store.add_documents(docs)
    print(f"Indexed {len(docs)} help-centre paragraphs from examples/sample_kb/")

    costs, metrics = CostTracker(budget=0.05), MetricsCollector()
    bot = AdaptiveRAGRetriever(
        vector_store=store,
        llm=ExtractiveStubLLM(),  # type: ignore[arg-type]  # swap for ChatOpenAI(...) etc.
        embedding_model=embeddings,
        fallback_strategies=[QueryVariationsStrategy(num_variations=2)],
        max_attempts=3,
        min_confidence=0.7,
        cost_tracker=costs,
        metrics_collector=metrics,
        enable_logging=False,
    )

    questions = [
        "How long do I have to get my money back?",
        "How many API tokens can one account have at the same time?",
    ]
    for question in questions:
        result = bot.query_with_fallback(question, return_intermediate_steps=True)
        print(f"\nQ: {question}")
        for step in result.intermediate_steps or []:
            print(
                f"   attempt {step['attempt']}: query={step['query']!r:<48} "
                f"answer={str(step.get('answer', '-'))[:38]!r} conf={step['confidence']:.2f}"
            )
        print(f"A: {result.answer}")
        print(f"   source={result.source}  confidence={result.confidence:.2f}")

    stats = metrics.get_stats()
    report = costs.get_report()
    print(
        f"\nqueries={stats['total_queries']} success_rate={stats['success_rate']:.0%} "
        f"avg_attempts={stats['avg_attempts']:.1f}"
    )
    print(
        f"estimated LLM spend=${report['total_cost']:.5f} "
        f"(budget used {report['budget_usage_percent']:.1f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
