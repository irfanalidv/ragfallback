"""Pre-flight and runtime diagnostics for RAG pipelines."""

from ragfallback.diagnostics.chunking import ChunkQualityChecker, ChunkQualityReport
from ragfallback.diagnostics.context_stitcher import OverlappingContextStitcher
from ragfallback.diagnostics.context_window import (
    CONTEXT_WINDOW_TOKEN_PRESETS,
    ContextWindowGuard,
    ContextWindowReport,
)
from ragfallback.diagnostics.embedding_guard import EmbeddingGuard, EmbeddingGuardReport
from ragfallback.diagnostics.embedding_probe import (
    EmbeddingQualityProbe,
    EmbeddingQualityReport,
)
from ragfallback.diagnostics.embedding_validator import (
    EmbeddingValidator,
    EmbeddingValidatorReport,
    infer_vectorstore_embedding_dim,
)
from ragfallback.diagnostics.retrieval_health import (
    RetrievalHealthCheck,
    RetrievalHealthReport,
)
from ragfallback.diagnostics.schema_sanitizer import (
    sanitize_documents,
    sanitize_metadata,
)
from ragfallback.diagnostics.stale_index import StaleIndexDetector, StaleIndexReport

__all__ = [
    "CONTEXT_WINDOW_TOKEN_PRESETS",
    "ChunkQualityChecker",
    "ChunkQualityReport",
    "ContextWindowGuard",
    "ContextWindowReport",
    "EmbeddingGuard",
    "EmbeddingGuardReport",
    "EmbeddingQualityProbe",
    "EmbeddingQualityReport",
    "EmbeddingValidator",
    "EmbeddingValidatorReport",
    "OverlappingContextStitcher",
    "RetrievalHealthCheck",
    "RetrievalHealthReport",
    "StaleIndexDetector",
    "StaleIndexReport",
    "infer_vectorstore_embedding_dim",
    "sanitize_documents",
    "sanitize_metadata",
]
