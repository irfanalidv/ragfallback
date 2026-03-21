# Architecture diagram specification

## Pipeline stages (left to right)
Ingest → Index → Retrieve → Generate → Evaluate

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

## Column 3 — ragfallback.strategies + ragfallback.core
Covers: Generate stage
Modules:
  AdaptiveRAGRetriever      (UC-6)
  QueryVariationsStrategy   (UC-6)
  MultiHopFallbackStrategy  (UC-6, multi_hop_demo)

## Column 4 — ragfallback.evaluation + ragfallback.tracking
Covers: Evaluate stage
Modules:
  RAGEvaluator      (UC-7)
  RAGScore          (UC-7)
  CostTracker       [used within UC-6]
  MetricsCollector  [used within UC-6]

## Known errors in current image (fix before regenerating)
1. StaleIndexDetector is labeled (UC-8) — WRONG. UC-8 is OverlappingContextStitcher.
   StaleIndexDetector has no dedicated UC example file.
2. MultiHopFallbackStrategy is labeled (UC-7) — WRONG. UC-7 is RAGEvaluator.
   MultiHopFallbackStrategy is part of UC-6 (uc6_multi_hop_demo.py).
3. Bottom-left problems box merges two separate problems into one bullet:
   "Query mismatch → silent empty results  corrupts index"
   These are two distinct problems. Correct:
     - Query mismatch → silent empty results  [UC-6]
     - Embedding model switch corrupts index   [UC-2]

## Install line
pip install ragfallback[chroma,huggingface]

## Version
v2.0.0
