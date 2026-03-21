"""
USE CASE: UC-1 — Retrieval health smoke test
=============================================
Real problem : Index quality degrades at scale; nobody probes retrieval before shipping.
Goal         : Run substring-based ``quick_check`` on a real persisted Chroma index.
Module       : ragfallback.diagnostics.RetrievalHealthCheck
Vector DB    : ChromaDB (local, no server)
Install      : pip install ragfallback[chroma,sentence-transformers]
Env vars     : NONE required
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import _kb_common
from ragfallback.diagnostics import RetrievalHealthCheck


def main() -> None:
    docs = _kb_common.load_sample_kb()
    persist = Path(tempfile.mkdtemp(prefix="ragfallback_uc1_")) / "chroma"
    vs = _kb_common.build_chroma_store(docs, persist_directory=persist, collection_name="uc1")
    health = RetrievalHealthCheck(k=4)
    report = health.quick_check(vs, docs, sample_size=min(10, len(docs)), seed=42)
    print(report.summary())
    if report.avg_latency_ms is not None:
        print(f"avg_latency_ms={report.avg_latency_ms:.1f}")
    print("hit_rate=", report.hit_rate)


if __name__ == "__main__":
    main()
