"""Tests for the Redis embedding cache."""
import numpy as np
import pytest
import redis
from unittest.mock import MagicMock

from processor.cache_handler import EmbeddingCache


def _make_cache():
    client = MagicMock(spec=redis.Redis)
    return EmbeddingCache(redis_client=client), client


class TestLookup:
    def test_miss_when_key_absent(self):
        cache, client = _make_cache()
        client.get.return_value = None

        result = cache.lookup("deadbeef")

        assert result.hit is False
        assert result.embedding is None
        assert cache.total_lookups == 1
        assert cache.cache_misses == 1
        assert cache.cache_hits == 0

    def test_hit_returns_stored_embedding(self):
        cache, client = _make_cache()
        original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        client.get.return_value = original.tobytes()

        result = cache.lookup("deadbeef")

        assert result.hit is True
        np.testing.assert_array_equal(result.embedding, original)
        assert cache.cache_hits == 1
        assert cache.cache_misses == 0

    def test_redis_error_counts_as_miss_not_a_crash(self):
        cache, client = _make_cache()
        client.get.side_effect = redis.RedisError("connection refused")

        result = cache.lookup("deadbeef")

        assert result.hit is False
        assert cache.cache_misses == 1

    def test_uses_prefixed_key(self):
        cache, client = _make_cache()
        client.get.return_value = None

        cache.lookup("deadbeef")

        client.get.assert_called_once_with("voyager:emb:deadbeef")


class TestStore:
    def test_store_calls_setex_with_ttl_and_bytes(self):
        cache, client = _make_cache()
        embedding = np.array([1.0, 2.0], dtype=np.float32)

        cache.store("deadbeef", embedding)

        client.setex.assert_called_once_with(
            "voyager:emb:deadbeef", cache._ttl, embedding.astype(np.float32).tobytes()
        )

    def test_store_swallows_redis_error(self):
        cache, client = _make_cache()
        client.setex.side_effect = redis.RedisError("write failed")

        cache.store("deadbeef", np.array([1.0], dtype=np.float32))  # must not raise


class TestHitRate:
    def test_zero_lookups_is_zero_rate(self):
        cache, _ = _make_cache()
        assert cache.hit_rate == 0.0

    def test_hit_rate_reflects_mixed_results(self):
        cache, client = _make_cache()
        client.get.side_effect = [b"\x00" * 4, None, b"\x00" * 4, None]

        for _ in range(4):
            cache.lookup("h")

        assert cache.hit_rate == 0.5
        assert cache.get_stats() == {
            "total_lookups": 4,
            "cache_hits": 2,
            "cache_misses": 2,
            "hit_rate": 0.5,
        }


class TestFlush:
    def test_flush_deletes_all_scanned_keys(self):
        cache, client = _make_cache()
        client.scan.side_effect = [(1, [b"voyager:emb:a", b"voyager:emb:b"]), (0, [])]

        cache.flush()

        assert client.delete.call_count == 1
        client.delete.assert_called_once_with(b"voyager:emb:a", b"voyager:emb:b")

    def test_flush_with_no_keys_does_not_call_delete(self):
        cache, client = _make_cache()
        client.scan.return_value = (0, [])

        cache.flush()

        client.delete.assert_not_called()
