# Contributing to ragfallback

Thank you for improving the library. This guide keeps contributions consistent with the architecture.

## Quick start

```bash
git clone https://github.com/irfanalidv/ragfallback
cd ragfallback
pip install -e ".[chroma,faiss,huggingface,hybrid]"
pip install -r requirements-dev.txt
pytest tests/unit/ -v   # must pass before any PR
```

## Adding a new module

Every new module needs:

- The correct subpackage (see table below)
- `from __future__ import annotations` at the top of every file
- Docstrings that explain *why*, not *what* — the code itself shows what
- `logging` instead of `print()` — library code never prints
- Unit tests in `tests/unit/test_{name}.py`
- An example file in `examples/` with a header block documenting the real problem, dataset, and install command
- `__all__` updated in the subpackage `__init__.py`

## Which subpackage?

| Your feature does… | Goes in |
|---------------------|---------|
| Runs before/during indexing | `ragfallback/diagnostics/` |
| Changes how retrieval works | `ragfallback/retrieval/` |
| Controls query retry logic | `ragfallback/strategies/` |
| Scores or observes after generation | `ragfallback/evaluation/` |
| Creates LLMs/embeddings/vector stores | `ragfallback/utils/` (factory only) |

## Coding standards (non-negotiable)

```python
# Every library file starts like this:
from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)      # not print()

class MyNewChecker:
    """One-line summary.

    Longer description if needed.

    Example::

        checker = MyNewChecker(threshold=0.8)
        report = checker.check(docs)
        print(report.summary())
    """

    def __init__(self, threshold: float = 0.8) -> None:
        """
        Args:
            threshold: Minimum score to pass. Must be in [0, 1].

        Raises:
            ValueError: If threshold is outside [0, 1].
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"threshold must be in [0, 1], got {threshold}. "
                "Pass threshold=0.8 for standard use."
            )
        self.threshold = threshold
```

## Use cases implemented

Each UC maps to one module and one example file.

| UC | Module | Status |
|----|--------|--------|
| UC-1 | `ragfallback.diagnostics.RetrievalHealthCheck` | ✅ done |
| UC-2 | `ragfallback.diagnostics.EmbeddingGuard` | ✅ done |
| UC-3 | `ragfallback.diagnostics.ChunkQualityChecker` | ✅ done |
| UC-4 | `ragfallback.diagnostics.ContextWindowGuard` | ✅ done |
| UC-5 | `ragfallback.retrieval.SmartThresholdHybridRetriever` | ✅ done |
| UC-6 | `ragfallback.core.AdaptiveRAGRetriever` | ✅ done |
| UC-7 | `ragfallback.evaluation.RAGEvaluator` | ✅ done |
| UC-8 | `ragfallback.diagnostics.OverlappingContextStitcher` | ✅ done |
| UC-9 | `ragfallback.diagnostics.EmbeddingQualityProbe` | ✅ done |
| UC-10 | `ragfallback.diagnostics.sanitize_documents` | ✅ done |

## PR checklist

- [ ] `pytest tests/unit/ -v` passes (all existing tests + new ones for your module)
- [ ] New module has type hints on all public methods
- [ ] No `print()` in library code — use `logging`
- [ ] No paid API keys in default paths
- [ ] `__all__` updated in subpackage `__init__.py`
- [ ] `CHANGELOG.md` `[Unreleased]` section updated with your addition
- [ ] Example file documents the real problem, dataset, and install command in its header block

## Running the full test suite

```bash
# Unit tests only (fast, no external deps)
pytest tests/unit/ -v

# Integration tests (requires chromadb + sentence-transformers)
pytest tests/integration/ -m integration -v

# Both
make test-all

# Run example scripts (skips ones needing Ollama/Docker/paid keys)
python run_all_examples.py
```

## Import rules

```python
# ✅ Correct: root exports (the 4 curated shortcuts)
from ragfallback import AdaptiveRAGRetriever, CostTracker

# ✅ Correct: subpackage imports for everything else
from ragfallback.diagnostics import ChunkQualityChecker, EmbeddingGuard
from ragfallback.retrieval import SmartThresholdHybridRetriever, FailoverRetriever
from ragfallback.strategies import MultiHopFallbackStrategy
from ragfallback.evaluation import RAGEvaluator

# ❌ Wrong: don't add new symbols to ragfallback/__init__.py
# unless they are genuinely the most-used entry points
```
