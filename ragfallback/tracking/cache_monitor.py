"""Cache hit/miss monitoring hook for RAG vector store queries."""

from __future__ import annotations

import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CacheEntry:
    """One cached retrieval result with hit tracking."""

    query_hash: str
    result: Any
    created_at: float
    hits: int = 0
    last_hit_at: Optional[float] = None


@dataclass
class CacheStats:
    """Snapshot of CacheMonitor statistics."""

    total_queries: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    avg_hit_latency_ms: float
    avg_miss_latency_ms: float
    cached_entries: int
    evictions: int

    def as_dict(self) -> Dict[str, Any]:
        """Return all fields as a plain JSON-serialisable dict."""
        return {
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": self.hit_rate,
            "avg_hit_latency_ms": self.avg_hit_latency_ms,
            "avg_miss_latency_ms": self.avg_miss_latency_ms,
            "cached_entries": self.cached_entries,
            "evictions": self.evictions,
        }


class CacheMonitor:
    """Transparent caching wrapper for any LangChain retriever.

    Wraps ``invoke`` / ``ainvoke`` to track hit rate, per-path latency,
    TTL expiry, and LRU eviction. Zero external dependencies — stdlib only.
    """

    def __init__(self, max_size: int = 256, ttl_seconds: float = 300.0) -> None:
        """Initialise the monitor with a maximum entry count and TTL.

        Args:
            max_size: Maximum number of cached entries before LRU eviction.
            ttl_seconds: Entries older than this (in seconds) are expired.
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._stats: Dict[str, float] = defaultdict(float)
        self.logger = logging.getLogger(__name__)

    # ── Hashing ────────────────────────────────────────────────────────────────

    def _hash_query(self, query: str, k: int) -> str:
        """Deterministic key for ``(query, k)`` pairs."""
        return hashlib.md5(
            f"{query.strip().lower()}:k={k}".encode()
        ).hexdigest()

    # ── Eviction ───────────────────────────────────────────────────────────────

    def _evict_expired(self) -> None:
        """Remove all entries that have exceeded their TTL."""
        now = time.monotonic()
        expired = [
            h for h, e in self._cache.items()
            if now - e.created_at > self._ttl
        ]
        for h in expired:
            del self._cache[h]
            self._stats["evictions"] += 1

    def _evict_lru(self) -> None:
        """Evict the least-recently-used entry when the cache is at capacity."""
        if len(self._cache) >= self._max_size:
            # Sort by last_hit_at (None → use created_at as fallback)
            lru_key = min(
                self._cache,
                key=lambda h: self._cache[h].last_hit_at or self._cache[h].created_at,
            )
            del self._cache[lru_key]
            self._stats["evictions"] += 1

    # ── Core logic (shared between sync and async wrappers) ────────────────────

    def _check_hit(self, query_hash: str) -> Optional[Any]:
        """Return cached result if present and unexpired, else ``None``."""
        entry = self._cache.get(query_hash)
        if entry is None:
            return None
        if time.monotonic() - entry.created_at > self._ttl:
            del self._cache[query_hash]
            self._stats["evictions"] += 1
            return None
        entry.hits += 1
        entry.last_hit_at = time.monotonic()
        return entry.result

    def _store(self, query_hash: str, result: Any) -> None:
        """Store a new result, evicting LRU if needed."""
        self._evict_lru()
        self._cache[query_hash] = CacheEntry(
            query_hash=query_hash,
            result=result,
            created_at=time.monotonic(),
        )

    # ── Public wrapper factory ─────────────────────────────────────────────────

    def wrap_retriever(self, retriever: Any, k: int = 4) -> Any:
        """Return a thin wrapper around ``retriever`` that caches results.

        The returned object exposes ``invoke(query)`` and ``ainvoke(query)``.
        All other attributes are transparently proxied to the original retriever.

        Args:
            retriever: Any LangChain retriever with ``invoke`` / ``ainvoke``.
            k: The ``k`` value used as part of the cache key.

        Returns:
            A wrapper object with the same interface as ``retriever``.
        """
        monitor = self

        class _CachedRetriever:
            """Proxy retriever with transparent hit/miss tracking."""

            def invoke(self, query: str) -> Any:
                """Sync invoke with caching."""
                t0 = time.monotonic()
                monitor._stats["total_queries"] += 1
                monitor._evict_expired()
                qh = monitor._hash_query(query, k)

                cached = monitor._check_hit(qh)
                if cached is not None:
                    elapsed = time.monotonic() - t0
                    monitor._stats["hits"] += 1
                    monitor._stats["hit_latency_total"] += elapsed
                    monitor.logger.debug("cache HIT  query=%s", query[:60])
                    return cached

                result = retriever.invoke(query)
                elapsed = time.monotonic() - t0
                monitor._stats["misses"] += 1
                monitor._stats["miss_latency_total"] += elapsed
                monitor._store(qh, result)
                monitor.logger.debug("cache MISS query=%s", query[:60])
                return result

            async def ainvoke(self, query: str) -> Any:
                """Async invoke with caching."""
                t0 = time.monotonic()
                monitor._stats["total_queries"] += 1
                monitor._evict_expired()
                qh = monitor._hash_query(query, k)

                cached = monitor._check_hit(qh)
                if cached is not None:
                    elapsed = time.monotonic() - t0
                    monitor._stats["hits"] += 1
                    monitor._stats["hit_latency_total"] += elapsed
                    monitor.logger.debug("cache HIT (async) query=%s", query[:60])
                    return cached

                ainvoke_fn = getattr(retriever, "ainvoke", None)
                if ainvoke_fn is not None:
                    result = await ainvoke_fn(query)
                else:
                    import asyncio
                    import functools
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, functools.partial(retriever.invoke, query)
                    )

                elapsed = time.monotonic() - t0
                monitor._stats["misses"] += 1
                monitor._stats["miss_latency_total"] += elapsed
                monitor._store(qh, result)
                monitor.logger.debug("cache MISS (async) query=%s", query[:60])
                return result

            def __getattr__(self, name: str) -> Any:
                return getattr(retriever, name)

        return _CachedRetriever()

    # ── Stats ──────────────────────────────────────────────────────────────────

    def get_stats(self) -> CacheStats:
        """Compute and return a :class:`CacheStats` snapshot."""
        total = int(self._stats["total_queries"])
        hits = int(self._stats["hits"])
        misses = int(self._stats["misses"])
        hit_rate = hits / total if total > 0 else 0.0
        avg_hit_ms = (
            self._stats["hit_latency_total"] / hits * 1000 if hits > 0 else 0.0
        )
        avg_miss_ms = (
            self._stats["miss_latency_total"] / misses * 1000 if misses > 0 else 0.0
        )
        return CacheStats(
            total_queries=total,
            cache_hits=hits,
            cache_misses=misses,
            hit_rate=hit_rate,
            avg_hit_latency_ms=avg_hit_ms,
            avg_miss_latency_ms=avg_miss_ms,
            cached_entries=len(self._cache),
            evictions=int(self._stats["evictions"]),
        )

    def reset(self) -> None:
        """Clear the cache and all statistics."""
        self._cache.clear()
        self._stats = defaultdict(float)

    def summary(self) -> str:
        """One-line human-readable cache summary."""
        s = self.get_stats()
        return (
            f"cache hit_rate={s.hit_rate:.1%} hits={s.cache_hits} "
            f"misses={s.cache_misses} entries={s.cached_entries} "
            f"evictions={s.evictions}"
        )
