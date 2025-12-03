# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

