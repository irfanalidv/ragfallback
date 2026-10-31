"""Normalize document metadata to JSON-safe, string-friendly values (glue / unhashable fixes)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document


def _normalize_value(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple, dict)):
        try:
            return json.dumps(value, default=str, ensure_ascii=False)
        except TypeError:
            return str(value)
    return str(value)


def sanitize_metadata(
    metadata: dict[str, Any],
) -> dict[str, str | int | float | bool | None]:
    """Flatten nested structures so filters / JSON / caches see stable scalar metadata."""
    out: dict[str, Any] = {}
    for k, v in (metadata or {}).items():
        key = str(k)
        if isinstance(v, dict):
            for nk, nv in v.items():
                out[f"{key}.{nk}"] = _normalize_value(nv)
        else:
            out[key] = _normalize_value(v)
    return out


def sanitize_documents(documents: Sequence[Document]) -> list[Document]:
    """Return new :class:`~langchain_core.documents.Document` instances with cleaned metadata."""
    result: list[Document] = []
    for doc in documents:
        md = sanitize_metadata(dict(doc.metadata or {}))
        result.append(Document(page_content=doc.page_content or "", metadata=md))
    return result
