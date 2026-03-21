# ragfallback

[![GitHub license](https://img.shields.io/github/license/irfanalidv/ragfallback)](https://github.com/irfanalidv/ragfallback/blob/main/LICENSE)
[![Python version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://pypi.org/project/ragfallback/)
[![PyPI](https://img.shields.io/pypi/v/ragfallback)](https://pypi.org/project/ragfallback/)
[![Downloads](https://static.pepy.tech/badge/ragfallback)](https://pepy.tech/project/ragfallback)
[![Tests](https://github.com/irfanalidv/ragfallback/actions/workflows/test.yml/badge.svg)](https://github.com/irfanalidv/ragfallback/actions/workflows/test.yml)

**ragfallback** prevents silent RAG failures across the full pipeline — from bad chunks at ingest, through retrieval outages at runtime, to invisible answer quality degradation in production.

![ragfallback architecture](ragfallback_architecture.png?v=2)

---

## What it prevents

| #   | Real production failure                               | Module                                             | Example                   |
| --- | ----------------------------------------------------- | -------------------------------------------------- | ------------------------- | --- |
| 1   | Query mismatch → silent empty results                 | `AdaptiveRAGRetriever` + `QueryVariationsStrategy` | `uc6_adaptive_rag.py`     |
| 2   | Embedding model switch corrupts index dimensions      | `EmbeddingGuard`                                   | `uc2_embedding_guard.py`  |
| 3   | Bad chunks (too short, mid-sentence) poison retrieval | `ChunkQualityChecker`                              | `uc3_chunk_quality.py`    |
| 4   | Retrieved chunks overflow LLM context window          | `ContextWindowGuard`                               | `uc4_context_window.py`   |
| 5   | Keyword queries fail dense retrieval silently         | `SmartThresholdHybridRetriever`                    | `uc5_hybrid_failover.py`  |
| 6   | Primary retriever outage returns empty, no fallback   | `FailoverRetriever`                                | `uc5_hybrid_failover.py`  |
| 7   | Multi-step questions always fail single-shot RAG      | `MultiHopFallbackStrategy`                         | `uc6_multi_hop_demo.py`   |
| 8   | Index serves stale data after document updates        | `StaleIndexDetector`                               | —                         |
| 9   | Answer quality invisible in production                | `RAGEvaluator`                                     | `uc7_rag_evaluator.py`    |
| 10  | Cross-boundary answers lost between adjacent chunks   | `OverlappingContextStitcher`                       | `uc8_context_stitcher.py` | ̋    |

---

## Quick start

```bash
pip install ragfallback[chroma,huggingface,real-data]
```

```python
# pip install ragfallback[chroma,huggingface,real-data]
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from ragfallback.diagnostics import ChunkQualityChecker, EmbeddingGuard, RetrievalHealthCheck
from ragfallback.evaluation import RAGEvaluator

# 1 — load 50 real Wikipedia passages (SQuAD, CC BY-SA 4.0)
ds = load_dataset("rajpurkar/squad", split="validation")
seen, docs, probes = set(), [], []
for row in ds:
    ctx = row["context"].strip()
    if ctx not in seen and len(seen) < 50:
        seen.add(ctx)
        docs.append(Document(page_content=ctx, metadata={"source": "squad"}))
    if row["answers"]["text"]:
        probes.append({"question": row["question"],
                       "ground_truth": row["answers"]["text"][0]})
print(f"Loaded {len(docs)} real passages, {len(probes)} Q&A pairs")

# 2 — check chunk quality before embedding
report = ChunkQualityChecker().check(docs)
print(report.summary())

# 3 — guard embedding dimensions before writing to any index
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
EmbeddingGuard(expected_dim=384).validate(embeddings).raise_if_failed()

# 4 — build index and smoke-test retrieval with real Q&A probes
store = Chroma.from_documents(docs, embeddings, persist_directory="./my_index")
health = RetrievalHealthCheck(k=4).run_substring_probes(
    store,
    {p["question"]: p["ground_truth"][:50] for p in probes[:10]},
)
print(f"Retrieval hit rate: {health.hit_rate:.0%}")

# 5 — evaluate answer quality on a real question
question = probes[0]["question"]
retrieved = store.as_retriever(search_kwargs={"k": 4}).invoke(question)
answer = retrieved[0].page_content if retrieved else "Not found"
score = RAGEvaluator().evaluate(
    question, answer,
    [d.page_content for d in retrieved],
    ground_truth=probes[0]["ground_truth"],
)
print(score.report())
```

Expected output (actual numbers — run it yourself):

```
Loaded 50 real passages, 2627 Q&A pairs
[PASS] chunks=50 | len min/avg/max=144/618/2095
Retrieval hit rate: 100%
========================================================
 RAG evaluation
========================================================
 Context precision  : 100.00%
 Faithfulness       : 95.00%
 Answer relevance   : 40.00%
 Recall (gold hit)  : 100.00%
 Overall            : 84.00%
 Pass (>=70%)       : True
```

---

## Configuration

Most features work with no API key — chunk checking, embedding validation, hybrid retrieval, and evaluation all run locally.

LLM-dependent features (`AdaptiveRAGRetriever`, `QueryVariationsStrategy`, `MultiHopFallbackStrategy`) need a model. Copy `.env.example` to `.env` and fill in:

```bash
cp .env.example .env
```

```
MISTRAL_API_KEY=your_key_here
MISTRAL_MODEL=mistral-small-latest   # default, override if needed
```

Get a free Mistral key at [console.mistral.ai](https://console.mistral.ai). The library also supports any LangChain-compatible LLM — pass it directly to `AdaptiveRAGRetriever(llm=your_llm)`.

---

## Full pipeline

```
Your documents
     │
     ▼
[ChunkQualityChecker]          ← bad splits, short/duplicate chunks
     │
     ▼
[EmbeddingGuard]               ← dimension / NaN / zero-vector checks before write
[EmbeddingQualityProbe]        ← domain mismatch heuristic (generic model on jargon)
[sanitize_documents]           ← JSON-safe metadata before any vector store write
     │
     ▼
Vector store (Chroma / FAISS / Qdrant / …)
     │
     ▼
[StaleIndexDetector]           ← SHA256 manifest: source files vs last build
     │
     ▼
[RetrievalHealthCheck]         ← labeled recall@k or quick substring smoke probes
     │
     ▼
[SmartThresholdHybridRetriever]  ← threshold + optional BM25 fallback
[FailoverRetriever]              ← primary → fallback on exception or empty results
     │
     ▼
[ContextWindowGuard]           ← rank + trim chunks to token budget (8 model presets)
[OverlappingContextStitcher]   ← merge adjacent chunks from same source
     │
     ▼
[AdaptiveRAGRetriever]         ← QueryVariationsStrategy / MultiHopFallbackStrategy
     │
     ▼
[RAGEvaluator]                 ← recall@k, nDCG, faithfulness (heuristic + LLM judge)
```

---

## Module reference

### `ragfallback.diagnostics`

**ChunkQualityChecker** — detects too-short, too-long, mid-sentence, and duplicate chunks before embedding.

```python
from ragfallback.diagnostics import ChunkQualityChecker
report = ChunkQualityChecker(min_chars=100, max_chars=8000).check(docs)
if report.has_issues:
    fixed = ChunkQualityChecker().auto_fix(docs)
```

**EmbeddingGuard** — validates dimension, NaN, and zero-vectors before writing to any index.

```python
from ragfallback.diagnostics import EmbeddingGuard
guard = EmbeddingGuard(expected_dim=384)
guard.validate(embeddings_model).raise_if_failed()        # model-level check
guard.validate_raw_vectors(vectors).raise_if_failed()     # pre-computed vectors
```

**EmbeddingQualityProbe** — heuristic domain-fit check: if similarity scores are uniformly low, the model is likely a poor domain match.

```python
from ragfallback.diagnostics import EmbeddingQualityProbe
result = EmbeddingQualityProbe().run(embeddings, query="...", reference_snippets=[...])
if not result.ok:
    print(result.warnings)   # "consider domain-specific model"
```

**RetrievalHealthCheck** — labeled recall@k or quick substring smoke probes against a live vector store.

```python
from ragfallback.diagnostics import RetrievalHealthCheck
health = RetrievalHealthCheck(k=5)
report = health.run_substring_probes(vector_store, {"What is Python?": "high-level language"})
print(report.hit_rate, report.avg_latency_ms)
```

**StaleIndexDetector** — SHA256 manifest to catch when source files changed since last index build.

```python
from ragfallback.diagnostics import StaleIndexDetector
det = StaleIndexDetector(manifest_path="./index_manifest.json")
det.record_paths(["./docs/policy.md"])          # record after build
report = det.check_paths(["./docs/policy.md"])  # check before serving
if report.has_stale:
    print(report.summary())
```

**ContextWindowGuard** — ranks and trims retrieved chunks to fit a token budget; 8 model presets included.

```python
from ragfallback.diagnostics import ContextWindowGuard
guard = ContextWindowGuard.from_model_name("gpt-4o")
selected, report = guard.select(query, retrieved_docs, embeddings)
```

**OverlappingContextStitcher** — merges consecutive chunks from the same source so cross-boundary answers aren't split.

```python
from ragfallback.diagnostics import OverlappingContextStitcher
merged = OverlappingContextStitcher().stitch(retrieved_docs)
```

**sanitize_documents** — normalizes list/dict/bytes metadata to JSON-safe scalars before any vector store write.

```python
from ragfallback.diagnostics import sanitize_documents
clean_docs = sanitize_documents(dirty_docs)   # safe for Chroma, Pinecone, Qdrant
```

---

### `ragfallback.retrieval`

**SmartThresholdHybridRetriever** — score-threshold gating with automatic BM25 fallback when dense scores are weak. Supports `distance`, `similarity`, and `relative` score modes.

```python
from ragfallback.retrieval import SmartThresholdHybridRetriever
retriever = SmartThresholdHybridRetriever.from_documents(
    docs, embeddings, dense_threshold=0.5, k=4
)   # pip install ragfallback[hybrid] for BM25
```

**FailoverRetriever** — if the primary retriever raises or returns fewer than `min_results` docs, automatically switches to a secondary.

```python
from ragfallback.retrieval import FailoverRetriever
retriever = FailoverRetriever(primary=chroma_retriever, fallback=faiss_retriever, min_results=1)
```

---

### `ragfallback.strategies`

**QueryVariationsStrategy** — LLM rewrites the original query into N variations to broaden retrieval recall.

**MultiHopFallbackStrategy** — decomposes complex multi-step questions into sub-questions, retrieves each independently, then synthesises a final answer.

```python
from ragfallback.strategies import MultiHopFallbackStrategy
result = MultiHopFallbackStrategy(max_hops=3).run(question, retriever, llm)
print(result.final_answer, result.total_hops)
```

---

### `ragfallback.evaluation`

**RAGEvaluator** — scores `recall@k`, `nDCG`, and `faithfulness` without external services. Optional LLM judge hook for higher accuracy.

```python
from ragfallback.evaluation import RAGEvaluator
ev = RAGEvaluator()
score = ev.evaluate(question, answer, context_docs, ground_truth="...")
print(score.overall_score, score.faithfulness_score, score.recall_at_k)
print(ev.batch_summary([score]))
```

---

## Examples — real public datasets

| Example                   | Dataset                                         | Command                                         |
| ------------------------- | ----------------------------------------------- | ----------------------------------------------- |
| UC-1: retrieval health    | SQuAD Wikipedia                                 | `python examples/uc1_retrieval_health.py`       |
| UC-2: embedding guard     | — (dimension check)                             | `python examples/uc2_embedding_guard.py`        |
| UC-3: chunk quality       | SQuAD Wikipedia                                 | `python examples/uc3_chunk_quality.py`          |
| UC-4: context window      | sample KB                                       | `python examples/uc4_context_window.py`         |
| UC-5: hybrid + failover   | FAISS + BM25                                    | `python examples/uc5_hybrid_failover.py`        |
| UC-6: adaptive RAG        | SQuAD Wikipedia (mock or Ollama LLM)            | `python examples/uc6_adaptive_rag.py`           |
| UC-7: RAG evaluator       | PubMedQA (MIT) — real medical Q&A               | `python examples/uc7_rag_evaluator.py`          |
| UC-8: context stitcher    | ChromaDB + HR chunks                            | `python examples/uc8_context_stitcher.py`       |
| UC-9: embedding probe     | — (similarity check)                            | `python examples/uc9_embedding_probe.py`        |
| UC-10: metadata sanitizer | ChromaDB dirty docs                             | `python examples/uc10_metadata_sanitizer.py`    |
| End-to-end on SQuAD       | SQuAD Wikipedia (CC BY-SA 4.0)                  | `python examples/real_data_demo.py`             |
| Financial news RAG        | nickmuchi/financial-classification (Apache 2.0) | `python examples/financial_risk_analysis.py`    |
| Legal contract RAG        | theatticusproject/cuad-qa (CC BY 4.0)           | `python examples/legal_document_analysis.py`    |
| Medical abstract RAG      | qiaojin/PubMedQA (MIT)                          | `python examples/medical_research_synthesis.py` |

---

## Verified numbers — SQuAD Wikipedia validation set

`python examples/real_data_demo.py` runs every module on 200 real Wikipedia passages. Numbers below are printed by the script on every run — not made up.

```
Passages indexed     : 200 real Wikipedia passages
Q&A pairs            : 10 570 (ground truth available)
ChunkQualityChecker  : 1 violation  (avg 662 chars/passage)
EmbeddingGuard       : OK — dim 384 matches expected 384

RetrievalHealthCheck (20 real Q&A substring probes):
  Hit rate   : 100.0%
  Avg latency: 25 ms per query

RAGEvaluator (10 real Q&A pairs, heuristic, no LLM judge):
  Pass rate        : 2/10  (heuristic; rises with LLM judge)
  Avg recall@k     : 100.0%
  Avg faithfulness : 79.5%
  Avg overall      : 62.9%
```

Install: `pip install ragfallback[chroma,huggingface,real-data]`  
Dataset: [rajpurkar/squad](https://huggingface.co/datasets/rajpurkar/squad) — CC BY-SA 4.0

---

## Install

```bash
pip install ragfallback                              # core only
pip install ragfallback[chroma,huggingface]          # golden path (no API keys)
pip install ragfallback[faiss,huggingface]           # FAISS instead of Chroma
pip install ragfallback[hybrid]                      # adds BM25 (rank_bm25)
pip install ragfallback[real-data]                   # real dataset examples (HuggingFace datasets)
```

| Extra         | Installs                               |
| ------------- | -------------------------------------- |
| `chroma`      | chromadb                               |
| `faiss`       | faiss-cpu                              |
| `huggingface` | sentence-transformers, huggingface-hub |
| `hybrid`      | rank_bm25, langchain-community         |
| `real-data`   | datasets                               |
| `openai`      | langchain-openai, openai               |

---

## Subpackage import map

```python
from ragfallback import AdaptiveRAGRetriever, QueryResult, CostTracker, MetricsCollector

from ragfallback.diagnostics import (
    ChunkQualityChecker, EmbeddingGuard, EmbeddingQualityProbe,
    RetrievalHealthCheck, StaleIndexDetector, ContextWindowGuard,
    OverlappingContextStitcher, sanitize_documents, sanitize_metadata,
)
from ragfallback.retrieval import SmartThresholdHybridRetriever, FailoverRetriever
from ragfallback.strategies import QueryVariationsStrategy, MultiHopFallbackStrategy
from ragfallback.evaluation import RAGEvaluator
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The quick version: run `pytest tests/unit/ -v` before any PR, follow Google-style docstrings, use `logging` not `print`, and update `__all__` in the subpackage `__init__.py`.

## License · Changelog

MIT License — see [LICENSE](LICENSE).  
Full version history in [CHANGELOG.md](CHANGELOG.md).
