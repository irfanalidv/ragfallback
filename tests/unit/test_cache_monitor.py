"""Unit tests for ragfallback.tracking.CacheMonitor."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, AsyncMock

import pytest

from ragfallback.tracking.cache_monitor import CacheMonitor, CacheStats


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_retriever(return_value=None):
    """Return a MagicMock retriever whose invoke() returns return_value."""
    r = MagicMock()
    r.invoke.return_value = return_value or ["doc1", "doc2"]
    return r


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestCacheMonitorMiss:
    """First-access behaviour (cache miss)."""

    def test_miss_calls_retriever(self):
        """Cache miss on first query — underlying retriever is called."""
        retriever = _make_retriever(["doc_a"])
        monitor = CacheMonitor()
        wrapped = monitor.wrap_retriever(retriever, k=4)
        result = wrapped.invoke("What is the refund policy?")
        assert result == ["doc_a"]
        retriever.invoke.assert_called_once()

    def test_miss_stores_result(self):
        """After a miss the result is stored in the internal cache."""
        monitor = CacheMonitor()
        wrapped = monitor.wrap_retriever(_make_retriever(["x"]), k=4)
        wrapped.invoke("unique query abc")
        assert monitor.get_stats().cached_entries == 1

    def test_miss_increments_misses(self):
        """Miss counter increases after the first call."""
        monitor = CacheMonitor()
        wrapped = monitor.wrap_retriever(_make_retriever())
        wrapped.invoke("q1")
        wrapped.invoke("q2")
        stats = monitor.get_stats()
        assert stats.cache_misses == 2
        assert stats.total_queries == 2


class TestCacheMonitorHit:
    """Repeat-access behaviour (cache hit)."""

    def test_hit_returns_same_result(self):
        """Identical query returns cached result without calling retriever again."""
        retriever = _make_retriever(["cached_doc"])
        monitor = CacheMonitor()
        wrapped = monitor.wrap_retriever(retriever, k=4)
        r1 = wrapped.invoke("same query")
        r2 = wrapped.invoke("same query")
        assert r1 == r2 == ["cached_doc"]
        assert retriever.invoke.call_count == 1  # only called once

    def test_hit_increments_hit_counter(self):
        """Hit count grows on each repeated call."""
        monitor = CacheMonitor()
        wrapped = monitor.wrap_retriever(_make_retriever())
        wrapped.invoke("repeat me")
        wrapped.invoke("repeat me")
        wrapped.invoke("repeat me")
        stats = monitor.get_stats()
        assert stats.cache_hits == 2
        assert stats.cache_misses == 1

    def test_different_queries_separate_entries(self):
        """Two different queries produce two distinct cache entries."""
        monitor = CacheMonitor()
        wrapped = monitor.wrap_retriever(_make_retriever())
        wrapped.invoke("query alpha")
        wrapped.invoke("query beta")
        assert monitor.get_stats().cached_entries == 2


class TestTTLEviction:
    """TTL-based expiry."""

    def test_ttl_expiry_triggers_miss(self):
        """After TTL expires the next call is treated as a miss."""
        monitor = CacheMonitor(ttl_seconds=0.05)  # 50 ms TTL
        wrapped = monitor.wrap_retriever(_make_retriever(["old"]))
        wrapped.invoke("ttl query")
        time.sleep(0.1)  # exceed TTL
        wrapped.invoke("ttl query")
        stats = monitor.get_stats()
        assert stats.cache_misses == 2   # both calls were misses
        assert stats.cache_hits == 0


class TestLRUEviction:
    """LRU max-size enforcement."""

    def test_lru_eviction_keeps_size_bounded(self):
        """Cache size never exceeds max_size."""
        max_size = 3
        monitor = CacheMonitor(max_size=max_size, ttl_seconds=3600)
        wrapped = monitor.wrap_retriever(_make_retriever())
        for i in range(max_size + 2):
            wrapped.invoke(f"unique query {i}")
        assert monitor.get_stats().cached_entries <= max_size

    def test_lru_eviction_increments_evictions(self):
        """Eviction counter increases when entries are displaced."""
        monitor = CacheMonitor(max_size=2, ttl_seconds=3600)
        wrapped = monitor.wrap_retriever(_make_retriever())
        wrapped.invoke("a")
        wrapped.invoke("b")
        wrapped.invoke("c")  # should evict one entry
        assert monitor.get_stats().evictions >= 1


class TestStats:
    """Statistics accuracy."""

    def test_hit_rate_correct(self):
        """hit_rate = hits / total_queries."""
        monitor = CacheMonitor()
        wrapped = monitor.wrap_retriever(_make_retriever())
        wrapped.invoke("q")       # miss
        wrapped.invoke("q")       # hit
        wrapped.invoke("q")       # hit
        stats = monitor.get_stats()
        assert abs(stats.hit_rate - 2 / 3) < 1e-9

    def test_hit_rate_zero_when_no_queries(self):
        """hit_rate is 0.0 before any queries."""
        monitor = CacheMonitor()
        assert monitor.get_stats().hit_rate == 0.0


class TestReset:
    """Reset clears state."""

    def test_reset_clears_cache_and_stats(self):
        """After reset() all stats are zero and cache is empty."""
        monitor = CacheMonitor()
        wrapped = monitor.wrap_retriever(_make_retriever())
        wrapped.invoke("something")
        wrapped.invoke("something")
        monitor.reset()
        stats = monitor.get_stats()
        assert stats.total_queries == 0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
        assert stats.cached_entries == 0


class TestSummary:
    """Human-readable summary."""

    def test_summary_non_empty_with_hit_rate(self):
        """summary() returns a non-empty string mentioning hit_rate."""
        monitor = CacheMonitor()
        wrapped = monitor.wrap_retriever(_make_retriever())
        wrapped.invoke("q1")
        wrapped.invoke("q1")  # hit
        s = monitor.summary()
        assert s
        assert "hit_rate" in s
        assert "hits" in s
        assert "misses" in s


class TestCacheStatsAsDict:
    """CacheStats.as_dict() serialisation."""

    def test_as_dict_contains_all_fields(self):
        """as_dict() includes every CacheStats field."""
        monitor = CacheMonitor()
        wrapped = monitor.wrap_retriever(_make_retriever())
        wrapped.invoke("x")
        d = monitor.get_stats().as_dict()
        expected_keys = {
            "total_queries", "cache_hits", "cache_misses", "hit_rate",
            "avg_hit_latency_ms", "avg_miss_latency_ms", "cached_entries", "evictions",
        }
        assert expected_keys == set(d.keys())
