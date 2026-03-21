"""
USE CASE: UC-1 + UC-6 — Chroma persisted KB + adaptive RAG
===========================================================
Real problem : User phrasing ≠ doc phrasing → empty retrieval; need real on-disk vectors.
Goal         : Load chunked ``sample_kb``, persist Chroma, run ``AdaptiveRAGRetriever`` with
               query variations using a **local** LLM (Ollama — no paid API keys).
Module       : ragfallback.core.AdaptiveRAGRetriever
Vector DB    : ChromaDB (local, no server)
Install      : pip install ragfallback[chroma,sentence-transformers]  &&  pip install -e . (from clone)
               Ollama: https://ollama.ai — ``ollama pull llama3`` (or set OLLAMA_MODEL)
Env vars     : NONE required (optional: OLLAMA_MODEL, OLLAMA_BASE_URL)
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

# Run from repo clone without ``pip install -e .``: prefer source tree on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if (_REPO_ROOT / "ragfallback").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

warnings.filterwarnings(
    "ignore",
    message=r".*doesn't match a supported version.*",
    category=Warning,
)

import _kb_common
from ragfallback import AdaptiveRAGRetriever, CostTracker, MetricsCollector
from ragfallback.utils.llm_factory import create_open_source_llm


def _run_demo() -> int:
    base = Path(__file__).resolve().parent
    persist_dir = str(base / ".chroma_kb_demo")

    print("=" * 70)
    print("ragfallback — Chroma + real KB markdown (community-issue shaped demo)")
    print("=" * 70)
    print(f"\nChroma persist: {persist_dir}")

    try:
        documents = _kb_common.load_sample_kb()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    if not documents:
        print("No chunks from sample_kb/", file=sys.stderr)
        return 1

    print(f"Loaded {len(documents)} chunk(s) from sample_kb.\n")

    try:
        embeddings = _kb_common.get_embeddings()
        vector_store = _kb_common.build_chroma_store(
            documents,
            persist_directory=persist_dir,
            collection_name="ragfallback_kb_demo",
        )
    except ImportError as e:
        print(f"{e}\nInstall: pip install chromadb sentence-transformers", file=sys.stderr)
        return 1

    print("Chroma collection ready (embeddings computed from real file contents).\n")

    model = os.environ.get("OLLAMA_MODEL", "llama3")
    base_url = os.environ.get("OLLAMA_BASE_URL")
    try:
        llm = create_open_source_llm(
            model=model,
            base_url=base_url,
            temperature=0,
            provider="ollama",
        )
    except ImportError as e:
        print(
            f"{e}\n"
            "Golden path uses local Ollama (no API keys).\n"
            "Install: https://ollama.ai  then: ollama pull llama3",
            file=sys.stderr,
        )
        return 1

    retriever = AdaptiveRAGRetriever(
        vector_store=vector_store,
        llm=llm,
        embedding_model=embeddings,
        fallback_strategy="query_variations",
        cost_tracker=CostTracker(),
        metrics_collector=MetricsCollector(),
        max_attempts=4,
        min_confidence=0.55,
    )

    question = (
        "How long until I see my money after a refund is approved, "
        "and what if I only watched a tiny bit of the course?"
    )

    print("Question (paraphrased / informal):")
    print(f"  {question}\n")
    print("Issue: plain similarity may miss 'refund' / 'business days' passages.\n")
    print("ragfallback: query variations + retries (LLM: Ollama).\n")
    print("-" * 70)

    try:
        result = retriever.query_with_fallback(
            question,
            return_intermediate_steps=True,
        )
    except Exception as e:
        err = str(e).lower()
        if "connection refused" in err or "11434" in err or "failed to establish" in err:
            print(
                "\n(Ollama not reachable — retrieval-only preview, no paid API keys.)\n"
                "For full adaptive RAG: install https://ollama.ai and run: ollama pull llama3\n"
            )
            hits = vector_store.similarity_search(question, k=4)
            for i, doc in enumerate(hits, 1):
                src = (doc.metadata or {}).get("source", "?")
                body = (doc.page_content or "")[:400].replace("\n", " ")
                print(f"  [{i}] source={src}\n      {body}{'…' if len(doc.page_content or '') > 400 else ''}\n")
            return 0
        print(
            "Adaptive RAG failed. Is Ollama running?\n"
            "  ollama serve   # or start the Ollama app\n"
            f"Detail: {e}",
            file=sys.stderr,
        )
        return 1

    print(f"\nAnswer:\n  {result.answer}\n")
    print(f"Confidence: {result.confidence:.2%} | attempts: {result.attempts} | cost: ${result.cost:.4f}")

    if result.intermediate_steps:
        print("\nIntermediate steps (queries tried):")
        for step in result.intermediate_steps:
            q = step.get("query", "")
            n = step.get("documents", 0)
            print(f"  • docs={n}  query={q[:100]}{'…' if len(q) > 100 else ''}")

    return 0


def main() -> int:
    import logging

    logging.getLogger("ragfallback.strategies.query_variations").setLevel(logging.CRITICAL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _run_demo()


if __name__ == "__main__":
    raise SystemExit(main())
