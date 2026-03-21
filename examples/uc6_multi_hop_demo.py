"""
USE CASE: UC-6 — Multi-hop retrieval for chained questions
==========================================================
Real problem : "What is the refund window for API keys that hit rate limits?" requires
               two separate retrieval hops — billing policy AND rate-limits docs.
               Single-shot RAG never reliably surfaces both pieces in one query.
Goal         : ``MultiHopFallbackStrategy.run()`` decomposes the question, retrieves
               evidence for each sub-question independently, then synthesises an answer.
Module       : ragfallback.strategies.MultiHopFallbackStrategy
Vector DB    : FAISS (in-memory, no server needed)
Install      : pip install ragfallback[sentence-transformers]
               Optional LLM: Ollama running locally (llama2 or any model)
               Without Ollama the script demonstrates mock-LLM mode automatically.
Env vars     : NONE required
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock

# Allow running directly from a repository clone without pip install -e .
_repo_root = Path(__file__).resolve().parent.parent
if (_repo_root / "ragfallback").is_dir() and str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import _kb_common  # noqa: E402
from ragfallback.strategies import HopResult, MultiHopFallbackStrategy, MultiHopResult  # noqa: E402


def _make_mock_llm() -> Any:
    """Return a scripted mock LLM that simulates multi-hop reasoning."""
    mock = MagicMock()
    responses = iter(
        [
            # Decompose
            '["What is the refund window for cancelled accounts?", '
            '"What rate limits apply to standard tier API keys?"]',
            # Hop 1 answer
            "Refunds are available within 30 days of the billing cycle for unused credits.",
            # Hop 2 answer
            "Standard tier API keys are limited to 60 requests per minute.",
            # Synthesis
            (
                "API keys that hit rate limits fall under the standard tier "
                "(60 requests/minute). If a user cancels after hitting limits, "
                "refunds on unused credits are available within 30 days of billing."
            ),
        ]
    )

    def _invoke(messages):
        return MagicMock(content=next(responses))

    mock.invoke.side_effect = _invoke
    return mock


def _try_ollama_llm() -> Any | None:
    """Attempt to connect to a locally running Ollama instance."""
    try:
        from langchain_community.llms import Ollama  # type: ignore

        llm = Ollama(model="llama2", timeout=5)
        llm.invoke("ping")  # quick connectivity check
        return llm
    except Exception:
        return None


def main() -> None:
    print("=" * 60)
    print("UC-6: Multi-hop retrieval demo")
    print("=" * 60)

    docs = _kb_common.load_sample_kb()
    emb = _kb_common.get_embeddings()
    vs = _kb_common.build_faiss_store(docs, emb)
    retriever = vs.as_retriever(search_kwargs={"k": 4})

    print(f"\nKnowledge base: {len(docs)} chunks loaded into FAISS.\n")

    llm = _try_ollama_llm()
    if llm is None:
        print(
            "[INFO] Ollama not reachable — using a scripted mock LLM.\n"
            "       To use a real LLM: start Ollama and run `ollama pull llama2`.\n"
        )
        llm = _make_mock_llm()
        mode = "mock"
    else:
        print("[INFO] Using Ollama (llama2).\n")
        mode = "live"

    question = (
        "What is the refund window for API keys that have hit rate limits?"
    )
    print(f"Question: {question}\n")

    strategy = MultiHopFallbackStrategy(max_hops=2, top_k=4)
    result: MultiHopResult = strategy.run(question, retriever=retriever, llm=llm)

    print(result.summary())
    print()

    for hop in result.hops:
        retrieved_preview = (
            hop.retrieved_chunks[0][:120] + "…"
            if hop.retrieved_chunks and len(hop.retrieved_chunks[0]) > 120
            else (hop.retrieved_chunks[0] if hop.retrieved_chunks else "(none)")
        )
        print(f"  Hop {hop.hop_number}: {hop.sub_question}")
        print(f"    Retrieved : {retrieved_preview}")
        print(f"    Answer    : {hop.partial_answer}")
        print(f"    Confidence: {hop.confidence:.2f}")
        print()

    print(f"Final answer ({mode} LLM):")
    print(f"  {result.final_answer}")
    print()

    # Also exercise generate_queries — the FallbackStrategy contract requires it
    print("-" * 60)
    print("FallbackStrategy contract — generate_queries(attempt=2):")
    mock_decompose_llm = _make_mock_llm()  # fresh responses
    sub_qs = strategy.generate_queries(question, context={}, attempt=2, llm=mock_decompose_llm)
    for i, q in enumerate(sub_qs, 1):
        print(f"  Sub-question {i}: {q}")

    print("\nDone.")


if __name__ == "__main__":
    main()
