# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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









