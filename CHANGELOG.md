# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.2.0] - 2026-04-05

### Added
- `AdaptiveRAGRetriever.aquery_with_fallback()` — native async retrieval.
  Mirrors `query_with_fallback()` exactly but uses LangChain `ainvoke()`
  throughout. Gracefully falls back to thread pool if `ainvoke` is not
  implemented by the underlying LLM or retriever. Enables true concurrent
  eval in `GoldenRunner.run_async()` and production FastAPI backends.
- `ragfallback.tracking.CacheMonitor` — hit/miss tracking wrapper for any
  LangChain retriever. Tracks hit rate, latency split by hit/miss, TTL-based
  expiry, LRU eviction, and entry count. Supports both sync `invoke()` and
  async `ainvoke()`. Exposes `get_stats() -> CacheStats`, `summary()`,
  and `reset()`. Exported from `ragfallback.tracking` and `ragfallback`.
- `CacheStats` dataclass — structured cache metrics: hit_rate, avg latencies,
  evictions. Includes `as_dict()` for JSON serialization.
- `GoldenReport.cache_stats` — optional field populated when `CacheMonitor`
  is passed to `GoldenRunner`. Surfaces cache efficiency alongside RAGAS
  and latency metrics in the same report.
- `GoldenRunner` optional `cache_monitor` param — pass a `CacheMonitor`
  instance to automatically wrap the retriever and capture cache stats
  during golden eval runs.

### Changed
- `GoldenRunner.run_async()` — now uses native `aquery_with_fallback()`
  instead of `run_in_executor` wrapping sync code. Falls back to thread
  pool automatically if retriever doesn't support `ainvoke`. This makes
  concurrent golden eval genuinely async — HTTP calls to LLM APIs now
  overlap instead of serializing through a thread pool.

### Fixed
- `tests/unit/test_async_retriever.py` — new test coverage for async path
- `tests/unit/test_cache_monitor.py` — new test coverage for CacheMonitor

## [2.1.0] - 2026-04-03

### Added

**`ragfallback/mlops/` — MLOps evaluation layer** (`pip install ragfallback[mlops]`)

- `RagasHook` — wraps RAGAS evaluation (faithfulness, answer relevance, context
  precision, context recall) as a native ragfallback hook. Degrades gracefully to
  heuristic evaluation if ragas is not installed — no crash, logged warning only.
- `RagasReport` — structured dataclass: all 4 RAGAS scores, `fallback_mode` flag,
  timestamp, raw output dict.
- `GoldenRunner` — async-native eval runner. Loads a golden dataset from JSON or
  `list[dict]`, runs `AdaptiveRAGRetriever` for each sample via `asyncio.gather`,
  tracks per-sample latency, fallback_triggered flag, and retrieved doc IDs.
  Computes recall@3, recall@5, P95 latency, mean latency, fallback rate.
- `GoldenReport` — structured dataclass with all `GoldenRunner` outputs.
- `BaselineRegistry` — JSON-backed metric store per dataset. `compare_or_fail()`
  raises `RegressionError` if any quality metric drops > threshold (default 5%)
  or latency P95 spikes > `latency_threshold` (default 12%). `update()` saves
  new baseline after a passing run.
- `RegressionError` — names which metrics regressed, old/new values, and delta.
- `QuerySimulator` — generates adversarial query mixes from a base query set:
  `short_keyword` (first 2 content words), `long_nl` (expand with instructions),
  `ambiguous` (strip proper nouns), `out_of_domain` (inject unrelated topic).
  `simulate_unhappy_paths()` produces all 4 types for every input query (4× expansion).
- `SimQuery` — dataclass: original, transformed, query_type.
- `MLflowLogger` — logs `GoldenReport` fields as MLflow metrics and params.
  No-op if mlflow is not installed.
- `generate_locustfile(output_path, endpoint)` — writes a ready-to-run Locust
  file simulating 4 realistic RAG query types with task weights matching real
  traffic distributions (short keyword 40%, long NL 20%, OOD 10%).
- `examples/build_golden_dataset.py` — builds `golden_qa.json` (75 SQuAD samples)
  and `golden_qa_stress.json` (25 SQuAD + 25 SciQ) from HuggingFace open data.
- `examples/mlops_demo.py` — full local demo: ChromaDB index → GoldenRunner →
  BaselineRegistry → MLflowLogger → QuerySimulator → generate_locustfile.
  Zero API keys required.
- `examples/ci_regression_gate.py` — CI-executable gate script. Exits 0 on pass,
  1 on regression. Uses `FakeListChatModel` (no API key).
- `examples/baselines.json` — committed baseline for `squad_ci` dataset.
- `examples/golden_qa.json`, `golden_qa_stress.json`, `golden_docs_registry.json`
  — committed so CI and new clones work without re-downloading HuggingFace datasets.
- `mlops` optional dependency group: `ragas>=0.2.0`, `mlflow>=2.10.0`,
  `locust>=2.20.0`, `aiohttp>=3.9.0`, `numpy>=1.24.0`.

### Fixed

- `recall_at_k` in `ragfallback/evaluation/rag_evaluator.py` — counts distinct
  relevant docs in the top-k slots so duplicates cannot push recall above 1.0.
- `BaselineRegistry.compare_or_fail` — accepts separate `latency_threshold`
  parameter (default 0.12) so CI can apply a looser gate on P95 latency
  (runner noise) vs. a strict 5% gate on quality metrics.

### Changed

- `.github/workflows/test.yml` — added `mlops-regression-gate` job: runs after
  unit tests pass, installs `ragfallback[mlops]`, builds golden dataset, runs
  `ci_regression_gate.py`, uploads `baselines.json` as a build artifact.

## [2.0.2] - 2026-03-22

### Changed
- Update PyPI package description to accurately reflect what the library does.

## [2.0.1] - 2026-03-22

### Fixed
- Replace deprecated `get_relevant_documents()` with `invoke()` across library source, examples, and tests — fixes compatibility with LangChain 0.2+.
- Prefer `langchain-huggingface` over deprecated `langchain-community` embeddings import; falls back gracefully on older installs.
- Probe HuggingFace and Ollama LLMs in the test fixture before returning them, so CI skips cleanly instead of failing when no LLM is reachable.

## [2.0.0] - 2026-03-21

### Breaking
- `ragfallback.__init__` now exports only 4 symbols: `AdaptiveRAGRetriever`, `QueryResult`, `CostTracker`, `MetricsCollector`. Use subpackage imports for everything else (e.g. `from ragfallback.diagnostics import ChunkQualityChecker`).

### Changed
- **Breaking:** `ragfallback.__init__` exports only `AdaptiveRAGRetriever`, `QueryResult`, `CostTracker`, `MetricsCollector` — use subpackages for diagnostics, retrieval, evaluation (see README)
- `pyproject.toml` optional extras: `huggingface` bundles sentence-transformers + transformers; `hybrid` includes `faiss-cpu`; `all` is open-source-only (no paid API stacks)
- README Examples: UC-1–UC-7 table with module and dataset columns

### Added
- `examples/_kb_common.py` — shared KB load + Chroma/FAISS builders (no `ragfallback` imports)
- `examples/uc1_retrieval_health.py` … `examples/uc7_rag_evaluator.py` — one script per canonical use case
- `examples/qdrant_local_demo.py` — Qdrant over Docker + thin vector shim (`qdrant-client` only)
- `tests/unit/` and `tests/integration/` — split test layout; `tests/integration/test_chroma_pipeline.py`; `Makefile` targets `test-unit` / `test-integration` / `test-all`
- `RetrievalHealthReport.avg_latency_ms` — populated for substring probes; removed duplicate `quick_check` definition
- `examples/production_reliability_example.py` — end-to-end pipeline (chunk → embed → Chroma → stale/content fingerprints → retrieval smoke → context budget → `RAGEvaluator`)
- `RAGScore`, `RAGEvaluator.evaluate` / `evaluate_batch` / `batch_summary` (heuristic or optional `llm=` judge)
- `StaleIndexDetector.build_content_manifest`, `record_document_contents`, `check_document_contents` (content SHA drift)
- `RetrievalHealthCheck.quick_check` (random substring probes)
- `ChunkQualityChecker` — `min_words`, mid-sentence stats, `auto_fix`
- `ContextWindowGuard` — optional `tiktoken_model`, `lost_in_middle` reorder in `select`
- `EmbeddingGuard.validate_raw_vectors` for pre-computed embedding matrices
- `docs/COMMON_RAG_ISSUES_AND_SOLUTIONS.md` — community RAG issue themes (API-sampled) mapped to library behavior
- `examples/sample_kb/` — markdown KB files on disk for realistic ingestion
- `examples/chroma_real_kb_demo.py` — persisted Chroma + paraphrased query demo
- `scripts/stackexchange_fetch_qa.py` — fetch question bodies + top answers via Stack Exchange API
- `research/README.md` — workflow for sampling and Q&A exports
- `ragfallback.diagnostics` — `ChunkQualityChecker`, `EmbeddingGuard`, `RetrievalHealthCheck`, `ContextWindowGuard`, `StaleIndexDetector`, `EmbeddingQualityProbe`
- `ragfallback.evaluation` — `RAGEvaluator` with `recall@k`, DCG/nDCG helpers, hooks for LLM-as-judge
- `MultiHopFallbackStrategy` — full multi-hop orchestration: decomposes question, retrieves per sub-question, synthesises final answer; `run(question, retriever, llm) -> MultiHopResult`; `generate_queries` satisfies `FallbackStrategy` contract (returns original on attempt 1, sub-questions on attempt ≥ 2)
- `MultiHopResult` dataclass — `final_answer`, `hops`, `total_hops`, `success`, `summary()`
- `HopResult` dataclass — `hop_number`, `sub_question`, `retrieved_chunks`, `partial_answer`, `confidence`
- `SmartThresholdHybridRetriever._dense_scores_are_weak(docs)` — public override point for the BM25-trigger decision; `from_documents` now raises `ImportError("BM25 hybrid requires: pip install ragfallback[hybrid]")` if `rank_bm25` is absent
- `FailoverRetriever.fallback=` — canonical parameter name (spec-aligned); legacy `secondary=` still accepted for backward compatibility
- `FailoverRetriever.min_results` — triggers failover when primary returns fewer than *min_results* documents (not just on exceptions); DEBUG logging records which retriever was used
- `tests/unit/test_multi_hop.py` — 23 unit tests for `MultiHopFallbackStrategy`
- `tests/unit/test_hybrid_retrieval.py` — 15 unit tests for Session 4 spec: `_dense_scores_are_weak`, BM25 path, `FailoverRetriever` on empty / exception / min_results, ImportError message, duck-typing checks
- `examples/uc6_multi_hop_demo.py` — UC-6 multi-hop demo with mock-LLM fallback (no Ollama required)
- `examples/uc5_hybrid_failover.py` — UC-5 demo: 4 scenarios (BM25 fallback, failover on empty, failover on exception, min_results)
- `docs/RELIABILITY_LAYERS.md` — five RAG pain points (semantic drift, dimension mismatch, redundant context, framework brittleness, observability) as reliability layers mapped to the library
- `ragfallback.retrieval` — `SmartThresholdHybridRetriever` (score threshold + BM25 fallback), `FailoverRetriever`, `ReRankerGuard`, `RetrieverAsVectorStore`
- `EmbeddingValidator`, `sanitize_documents` / `sanitize_metadata`, `OverlappingContextStitcher`
- Optional dependency group `hybrid` (`rank_bm25`) in `pyproject.toml`
- README: full-pipeline reliability section (accurate APIs vs LangChain comparison table)
- `ChunkQualityReport.has_issues`, `ChunkQualityChecker.suggest_fixes`, `StaleIndexReport.has_stale`
- `StaleIndexDetector.record_paths`, `check_paths`, `record_from_documents`
- `RetrievalHealthCheck.run_substring_probes`, `RetrievalHealthReport.hit_rate`
- `EmbeddingGuardReport.raise_if_failed`, `EmbeddingDimensionError`, row sanity checks in `EmbeddingGuard`
- `CONTEXT_WINDOW_TOKEN_PRESETS`, `ContextWindowGuard.from_model_name`, `ContextWindowReport.summary`
- `RAGEvaluator.evaluate_heuristic`, `SimpleRAGReport`

## [0.1.0] - 2024-12-03

### Added
- Initial release of ragfallback library
- Core `AdaptiveRAGRetriever` with fallback strategies
- `QueryVariationsStrategy` for LLM-based query rewriting
- `CostTracker` for token cost tracking and budget management
- `MetricsCollector` for performance metrics
- `ConfidenceScorer` for answer quality assessment
- Factory functions for LLMs, embeddings, and vector stores
- Support for open-source components (HuggingFace, Ollama, FAISS)
- Support for paid providers (OpenAI, Anthropic, Pinecone)
- 11 comprehensive examples including:
  - Basic usage examples
  - Advanced use cases (legal, medical, financial analysis)
  - Multi-domain knowledge synthesis
- Complete test suite with integration tests
- Documentation and installation guides

### Features
- Query fallback with automatic retries
- Confidence-based answer validation
- Budget enforcement and cost alerts
- Metrics collection and reporting
- Flexible configuration options
- Production-ready error handling and logging









