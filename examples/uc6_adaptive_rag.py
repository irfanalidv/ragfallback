"""
USE CASE: UC-6 — Adaptive RAG with query variations fallback
=============================================================
Real problem : User asks "how do I cancel?" — document says "subscription termination".
               Zero retrieval results. Standard RAG returns "I don't know" silently.
Goal         : AdaptiveRAGRetriever retries with LLM-generated query rewrites until it
               finds the answer. Works with Ollama (free, local) or a scripted mock LLM
               if no LLM is available, so the retry mechanism is always visible.
Dataset      : SQuAD Wikipedia — 100 real passages (CC BY-SA 4.0)
               https://huggingface.co/datasets/rajpurkar/squad
Module       : ragfallback.core.AdaptiveRAGRetriever
Vector DB    : FAISS (local, no server)
Install      : pip install ragfallback[faiss,huggingface,real-data]
               Optional LLM: ollama pull llama3
Env vars     : NONE required (mock LLM used automatically when Ollama is absent)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

_repo_root = Path(__file__).resolve().parent.parent
if (_repo_root / "ragfallback").is_dir() and str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

_examples_dir = Path(__file__).resolve().parent
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

import _kb_common  # noqa: E402
from ragfallback import AdaptiveRAGRetriever  # noqa: E402
from ragfallback.strategies import QueryVariationsStrategy  # noqa: E402
from ragfallback.tracking import CostTracker, MetricsCollector  # noqa: E402


# Pre-scripted variations for three demo queries. The mock cycles through
# them deterministically so the retry chain is always visible even offline.
_MOCK_VARIATIONS: dict[int, list[str]] = {
    0: [
        '["What is the birthplace of Nikola Tesla?", "Where was Nikola Tesla born?"]',
        "Nikola Tesla was born in the village of Smiljan.",
    ],
    1: [
        '["What did Marie Curie discover?", "What Nobel Prize did Marie Curie win?"]',
        "Marie Curie discovered polonium and radium.",
    ],
    2: [
        '["How long is the Great Wall of China?", "What is the length of the Great Wall?"]',
        "The Great Wall of China stretches over 13,000 miles.",
    ],
}

_mock_call_index = 0


def _make_mock_llm() -> Any:
    """Return a scripted mock that cycles through pre-canned variations."""
    mock = MagicMock()
    call_counter: list[int] = [0]

    def _invoke(messages):
        idx = call_counter[0] % 6
        query_group = idx // 2
        response_slot = idx % 2
        call_counter[0] += 1
        responses = _MOCK_VARIATIONS.get(query_group, _MOCK_VARIATIONS[0])
        return MagicMock(content=responses[response_slot])

    mock.invoke.side_effect = _invoke
    return mock


def _try_ollama_llm() -> Any | None:
    try:
        from langchain_community.llms import Ollama  # type: ignore

        llm = Ollama(model="llama3", timeout=3)
        llm.invoke("ping")
        return llm
    except Exception:
        return None


# Queries chosen to exercise the variation retry path — phrasing deliberately
# mismatches the Wikipedia text to show that the first attempt will fail.
_DEMO_QUERIES = [
    {
        "question": "Where was the inventor of AC electricity born?",
        "note": "Phrasing uses 'inventor of AC electricity', docs mention 'Nikola Tesla'",
    },
    {
        "question": "Which elements did the first female Nobel laureate discover?",
        "note": "Phrasing uses 'first female Nobel laureate', docs mention 'Marie Curie'",
    },
    {
        "question": "What is the distance spanned by the ancient Chinese defensive wall?",
        "note": "Phrasing uses 'ancient Chinese defensive wall', docs say 'Great Wall'",
    },
]


def main() -> None:
    print("=" * 66)
    print("UC-6: AdaptiveRAGRetriever — query variation retry demo")
    print("Dataset: SQuAD Wikipedia (CC BY-SA 4.0)")
    print("=" * 66)

    print("\nLoading 100 real SQuAD Wikipedia passages...")
    try:
        docs, probes = _kb_common.load_real_dataset("squad", max_docs=100)
    except (ImportError, Exception) as exc:
        print(f"  WARN: {exc} — falling back to sample_kb")
        docs = _kb_common.load_sample_kb()
        probes = []

    print(f"\nBuilding FAISS index ({len(docs)} passages)...")
    emb = _kb_common.get_embeddings()
    try:
        from langchain_community.vectorstores import FAISS
    except ImportError:
        print("SKIP: pip install ragfallback[faiss] to run this example")
        sys.exit(0)

    vs = FAISS.from_documents(docs, emb)

    llm = _try_ollama_llm()
    if llm is not None:
        mode = "Ollama (real LLM)"
    else:
        llm = _make_mock_llm()
        mode = "Mock LLM (scripted — Ollama not available)"
    print(f"\nLLM mode: {mode}")

    adaptive = AdaptiveRAGRetriever(
        vector_store=vs,
        llm=llm,
        embedding_model=emb,
        fallback_strategies=[QueryVariationsStrategy(num_variations=2)],
        cost_tracker=CostTracker(),
        metrics_collector=MetricsCollector(),
        max_attempts=6,
        min_confidence=0.30,
    )

    print("\n" + "─" * 66)
    print("Query variation retry demo — three queries with mismatched phrasing")
    print("─" * 66)

    for i, demo in enumerate(_DEMO_QUERIES, 1):
        question = demo["question"]
        print(f"\nQuery {i}: {question}")
        print(f"  Why: {demo['note']}")

        result = adaptive.query_with_fallback(
            question=question,
            return_intermediate_steps=True,
        )

        if result.intermediate_steps:
            for step in result.intermediate_steps:
                attempt = step.get("attempt", "?")
                query = step.get("query", "")[:70]
                conf = step.get("confidence", 0.0)
                print(f"  → attempt {attempt}: '{query}...'  confidence={conf:.0%}")

        print(f"  Final answer : {(result.answer or 'Not found')[:120]}")
        print(f"  Succeeded at : attempt {result.attempts}  confidence={result.confidence:.0%}")
        if result.cost == 0.0:
            print(f"  Cost         : $0.0000 (mock LLM — no real inference)")
        else:
            print(f"  Cost         : ${result.cost:.4f}")

    stats = adaptive.metrics_collector.get_stats()
    print("\n" + "─" * 66)
    print(f"Session stats: {stats['total_queries']} queries  "
          f"success_rate={stats['success_rate']:.0%}  "
          f"avg_confidence={stats['avg_confidence']:.0%}")

    if llm.__class__.__name__ == "MagicMock":
        print(
            "\nNote: Running in mock mode. The retry chain above is scripted to show"
            "\nthe mechanism. Pass `ollama pull llama3` and restart to see real variation."
        )

    print("\n✅  UC-6 demo complete.")


if __name__ == "__main__":
    main()
