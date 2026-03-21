"""Merge adjacent chunks from the same source to recover cross-boundary context."""

from __future__ import annotations

from typing import List, Optional, Sequence

from langchain_core.documents import Document


def _parse_int(val: Optional[object]) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


class OverlappingContextStitcher:
    """
    If metadata marks chunk order (e.g. ``chunk_index``) and ``source``, merge consecutive
    chunks from one source into one larger block for prompting.

    Also provides :meth:`expand_neighbor_indices` to widen around a single index (window).
    """

    def __init__(
        self,
        chunk_index_key: str = "chunk_index",
        source_key: str = "source",
        separator: str = "\n\n",
    ):
        self.chunk_index_key = chunk_index_key
        self.source_key = source_key
        self.separator = separator

    def stitch(self, documents: Sequence[Document]) -> List[Document]:
        if not documents:
            return []

        buckets: dict = {}
        for doc in documents:
            src = (doc.metadata or {}).get(self.source_key, "_unknown")
            idx = _parse_int((doc.metadata or {}).get(self.chunk_index_key))
            buckets.setdefault(src, []).append((idx, doc))

        merged: List[Document] = []
        for src, items in buckets.items():
            items.sort(key=lambda x: (x[0] is None, x[0] if x[0] is not None else 0))
            run: List[Document] = []
            last_idx: Optional[int] = None

            def flush_run(run_docs: List[Document]) -> None:
                if not run_docs:
                    return
                text = self.separator.join(d.page_content for d in run_docs)
                base_md = dict(run_docs[0].metadata or {})
                base_md[self.source_key] = src
                if _parse_int(base_md.get(self.chunk_index_key)) is not None:
                    base_md["stitched_chunk_start"] = base_md.get(self.chunk_index_key)
                    base_md["stitched_chunk_end"] = (run_docs[-1].metadata or {}).get(
                        self.chunk_index_key
                    )
                base_md["ragfallback_stitched"] = True
                merged.append(Document(page_content=text, metadata=base_md))

            for idx, doc in items:
                if idx is None:
                    flush_run(run)
                    run = []
                    merged.append(doc)
                    last_idx = None
                    continue
                if last_idx is None or idx == last_idx + 1:
                    run.append(doc)
                    last_idx = idx
                else:
                    flush_run(run)
                    run = [doc]
                    last_idx = idx
            flush_run(run)

        return merged

    def expand_neighbor_indices(
        self,
        documents: Sequence[Document],
        neighbor_radius: int = 1,
    ) -> List[Document]:
        """
        Given retrieved docs with ``chunk_index`` + ``source``, add synthetic docs is not
        supported without the full corpus — returns input unchanged if indices missing.

        When all docs share source and have indices, duplicate single-hit windows by
        including min-1..max+1 range only among *already present* docs (no corpus fetch).
        """
        if neighbor_radius <= 0 or not documents:
            return list(documents)

        by_src: dict = {}
        for d in documents:
            md = d.metadata or {}
            src = md.get(self.source_key)
            idx = _parse_int(md.get(self.chunk_index_key))
            if src is None or idx is None:
                return list(documents)
            by_src.setdefault(src, []).append((idx, d))

        out: List[Document] = []
        for src, items in by_src.items():
            indices = sorted(set(i for i, _ in items))
            want = set()
            for i in indices:
                for j in range(i - neighbor_radius, i + neighbor_radius + 1):
                    want.add(j)
            picked = [(i, d) for i, d in items if i in want]
            picked.sort(key=lambda x: x[0])
            out.extend(d for _, d in picked)
        return out if out else list(documents)
