"""Detect when source files changed since the last index build (manifest + checksums)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class StaleIndexReport:
    ok: bool
    stale_paths: List[str] = field(default_factory=list)
    missing_paths: List[str] = field(default_factory=list)
    new_paths: List[str] = field(default_factory=list)
    message: str = ""

    @property
    def has_stale(self) -> bool:
        """True if the manifest disagrees with current files (same as ``not ok`` when manifest existed)."""
        return not self.ok

    def summary(self) -> str:
        return self.message or (
            f"stale_index ok={self.ok} stale={len(self.stale_paths)} "
            f"missing={len(self.missing_paths)} new={len(self.new_paths)}"
        )


class StaleIndexDetector:
    """
    Compare current sources to a saved manifest (path -> sha256). Use after ingest to
    persist ``manifest.json`` alongside the vector store; reload before queries in production.
    """

    def __init__(self, manifest_path: Optional[Union[str, Path]] = None):
        self.manifest_path = Path(manifest_path) if manifest_path else None

    def build_manifest(self, paths: Iterable[Union[str, Path]]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for p in paths:
            path = Path(p)
            if path.is_file():
                out[str(path.resolve())] = _sha256_file(path)
        return out

    def save_manifest(self, mapping: Dict[str, str], path: Optional[Union[str, Path]] = None) -> Path:
        target = Path(path or self.manifest_path or "index_manifest.json")
        target.write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")
        return target

    def load_manifest(self, path: Optional[Union[str, Path]] = None) -> Dict[str, str]:
        src = Path(path or self.manifest_path or "index_manifest.json")
        if not src.is_file():
            return {}
        return json.loads(src.read_text(encoding="utf-8"))

    def diff(
        self,
        current_paths: Iterable[Union[str, Path]],
        saved: Optional[Dict[str, str]] = None,
    ) -> StaleIndexReport:
        saved = saved if saved is not None else self.load_manifest()
        current: Dict[str, str] = {}
        for p in current_paths:
            path = Path(p)
            key = str(path.resolve())
            if path.is_file():
                current[key] = _sha256_file(path)

        stale: List[str] = []
        missing: List[str] = []
        new_paths: List[str] = []

        for key, old_hash in saved.items():
            if key not in current:
                missing.append(key)
            elif current[key] != old_hash:
                stale.append(key)

        for key in current:
            if key not in saved:
                new_paths.append(key)

        ok = not stale and not missing and not new_paths and bool(saved)
        if not saved:
            msg = "No saved manifest — cannot detect staleness (save one at index time)."
            return StaleIndexReport(ok=False, message=msg)
        if ok:
            msg = "Manifest matches current source files."
        else:
            parts = []
            if stale:
                parts.append(f"{len(stale)} changed")
            if missing:
                parts.append(f"{len(missing)} removed")
            if new_paths:
                parts.append(f"{len(new_paths)} new")
            msg = "Index may be stale: " + ", ".join(parts) + "."
        return StaleIndexReport(
            ok=ok,
            stale_paths=stale,
            missing_paths=missing,
            new_paths=new_paths,
            message=msg,
        )

    def record_paths(
        self,
        paths: Iterable[Union[str, Path]],
        manifest_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Fingerprint files and save manifest (call after a successful index build)."""
        m = self.build_manifest(paths)
        return self.save_manifest(m, path=manifest_path or self.manifest_path)

    def check_paths(
        self,
        paths: Iterable[Union[str, Path]],
        manifest_path: Optional[Union[str, Path]] = None,
    ) -> StaleIndexReport:
        """Compare live files to saved manifest (e.g. on worker startup)."""
        saved = self.load_manifest(manifest_path or self.manifest_path)
        return self.diff(paths, saved=saved)

    def record_from_documents(
        self,
        documents: Sequence[Any],
        path_key: str = "path",
        source_key: str = "source",
        manifest_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Build manifest from document metadata paths. Uses ``path_key`` then ``source_key``
        when they point to existing files on disk.
        """
        paths: List[Path] = []
        for doc in documents:
            md = getattr(doc, "metadata", None) or {}
            for key in (path_key, source_key):
                if key in md and md[key]:
                    p = Path(str(md[key]))
                    if p.is_file():
                        paths.append(p)
                        break
        return self.record_paths(paths, manifest_path=manifest_path)

    def build_content_manifest(
        self,
        documents: Sequence[Any],
        id_key: str = "source",
    ) -> Dict[str, str]:
        """
        SHA-256 of each chunk's ``page_content``, keyed by ``metadata[id_key]`` and index
        (for in-memory-only corpora with no file paths).
        """
        out: Dict[str, str] = {}
        for i, doc in enumerate(documents):
            body = (getattr(doc, "page_content", "") or "").encode("utf-8")
            md = getattr(doc, "metadata", None) or {}
            base = str(md.get(id_key, f"chunk_{i}"))
            key = f"{base}::{i}"
            out[key] = hashlib.sha256(body).hexdigest()
        return out

    def record_document_contents(
        self,
        documents: Sequence[Any],
        manifest_path: Optional[Union[str, Path]] = None,
        id_key: str = "source",
    ) -> Path:
        """Save content fingerprints after indexing (when sources are not file paths)."""
        m = self.build_content_manifest(documents, id_key=id_key)
        return self.save_manifest(m, path=manifest_path or self.manifest_path)

    def check_document_contents(
        self,
        documents: Sequence[Any],
        manifest_path: Optional[Union[str, Path]] = None,
        id_key: str = "source",
    ) -> StaleIndexReport:
        """Compare current in-memory chunks to the last saved content manifest."""
        saved = self.load_manifest(manifest_path or self.manifest_path)
        current = self.build_content_manifest(documents, id_key=id_key)
        return self._diff_maps(saved, current)

    def _diff_maps(self, saved: Dict[str, str], current: Dict[str, str]) -> StaleIndexReport:
        stale: List[str] = []
        missing: List[str] = []
        new_paths: List[str] = []
        for key, old_h in saved.items():
            if key not in current:
                missing.append(key)
            elif current[key] != old_h:
                stale.append(key)
        for key in current:
            if key not in saved:
                new_paths.append(key)
        ok = not stale and not missing and not new_paths and bool(saved)
        if not saved:
            return StaleIndexReport(
                ok=False,
                message="No saved manifest — cannot detect content drift.",
            )
        if ok:
            msg = "Content manifest matches indexed chunks."
        else:
            parts = []
            if stale:
                parts.append(f"{len(stale)} changed")
            if missing:
                parts.append(f"{len(missing)} removed")
            if new_paths:
                parts.append(f"{len(new_paths)} new")
            msg = "Indexed content may be stale: " + ", ".join(parts) + "."
        return StaleIndexReport(
            ok=ok,
            stale_paths=stale,
            missing_paths=missing,
            new_paths=new_paths,
            message=msg,
        )
