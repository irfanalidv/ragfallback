# Architecture diagram specification

Source of truth for `ragfallback_arch.svg`. Regenerate from this spec
whenever a module is added, renamed, or moved between subpackages —
keep the diagram and `README.md`'s module reference in sync.

## Pipeline stages (left to right)
Ingest → Index → Retrieve → Generate → Evaluate → Operate

`Operate` represents the MLOps loop (golden-set eval, regression gate,
load testing) running continuously against a deployed pipeline — it
feeds back into all five earlier stages, not just `Evaluate`.

## Column 1 — ragfallback.diagnostics
Covers: Ingest + Index stages
Modules:
  ChunkQualityChecker    (UC-3)
  EmbeddingGuard         (UC-2)
  EmbeddingQualityProbe  (UC-9)
  sanitize_documents     (UC-10)
  StaleIndexDetector     [no UC example — inline use]
  RetrievalHealthCheck   (UC-1)
  ContextWindowGuard     (UC-4)
  OverlappingContextStitcher (UC-8)

## Column 2 — ragfallback.retrieval
Covers: Retrieve stage
Modules:
  SmartThresholdHybridRetriever (UC-5)
  FailoverRetriever             (UC-5)
  ReRankerGuard                 [hook, no dedicated example]
  RetrieverAsVectorStore        [shim, no dedicated example]

## Column 3 — ragfallback.core + ragfallback.strategies
Covers: Generate stage
Modules:
  AdaptiveRAGRetriever          (UC-6)
  AdaptiveRAGRetriever.aquery_with_fallback() [async, added v2.2.0]
  QueryVariationsStrategy       (UC-6)
  MultiHopFallbackStrategy      (UC-6, multi_hop_demo)

## Column 4 — ragfallback.evaluation + ragfallback.tracking
Covers: Evaluate stage
Modules:
  RAGEvaluator      (UC-7)
  CostTracker       [used within UC-6]
  MetricsCollector  [used within UC-6]
  CacheMonitor      [hit/miss, added v2.2.0]

## Row 5 — ragfallback.mlops (full-width, "Operate" stage)
  GoldenRunner       — async eval loop: recall@k, latency P95, fallback rate
  BaselineRegistry   — compare_or_fail() raises RegressionError on >5% drop
  RagasHook          — RAGAS when installed, heuristic fallback otherwise
  QuerySimulator     — short_keyword / long_nl / ambiguous / out_of_domain
  MLflowLogger + generate_locustfile()

## Install line
pip install ragfallback[chroma,huggingface]

## Version
v2.2.1

## History
- v2.0.2 image had two labeling bugs (StaleIndexDetector mislabeled UC-8,
  MultiHopFallbackStrategy mislabeled UC-7) and was pulled from the README
  rather than shipped wrong. Fixed in the v2.2.1 regeneration — see
  CHANGELOG.md.
- v2.2.1 switched the asset from PNG to SVG (crisp at any zoom, ~9KB vs
  ~1.2MB) and added the mlops row, which didn't exist as a subpackage
  when the original diagram was drawn.
