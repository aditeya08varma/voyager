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
        # Two hashes 64 bits apart so a miss on one never fuzzy-matches the other.
        client.get.side_effect = [b"\x00" * 4, None, b"\x00" * 4, None]

        cache.lookup("0000000000000000")
        cache.lookup("ffffffffffffffff")
        cache.lookup("0000000000000000")
        cache.lookup("ffffffffffffffff")

        assert cache.hit_rate == 0.5
        assert cache.get_stats() == {
            "total_lookups": 4,
            "cache_hits": 2,
            "cache_misses": 2,
            "fuzzy_hits": 0,
            "hit_rate": 0.5,
        }


class TestFuzzyMatching:
    def test_close_hash_reuses_recent_embedding(self):
        cache, client = _make_cache()
        original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        # First frame: exact hit, seeds the fuzzy window.
        client.get.return_value = original.tobytes()
        cache.lookup("0000000000000000")

        # Second frame: 2 bits different (within the 3-bit threshold), Redis
        # has never seen this exact hash.
        client.get.return_value = None
        result = cache.lookup("0000000000000003")

        assert result.hit is True
        assert result.fuzzy is True
        np.testing.assert_array_equal(result.embedding, original)
        assert cache.fuzzy_hits == 1
        assert cache.cache_misses == 0

    def test_hash_beyond_threshold_stays_a_real_miss(self):
        cache, client = _make_cache()
        client.get.return_value = np.array([1.0], dtype=np.float32).tobytes()
        cache.lookup("0000000000000000")

        # Far outside the 3-bit threshold.
        client.get.return_value = None
        result = cache.lookup("ffffffffffffffff")

        assert result.hit is False
        assert result.fuzzy is False
        assert cache.cache_misses == 1
        assert cache.fuzzy_hits == 0

    def test_fuzzy_window_evicts_oldest_beyond_capacity(self):
        cache, client = _make_cache()
        client.get.return_value = None

        for i in range(cache.FUZZY_WINDOW_SIZE + 10):
            cache.store(f"{i:016x}", np.array([1.0], dtype=np.float32))

        assert len(cache._recent_hashes) == cache.FUZZY_WINDOW_SIZE
        # The oldest entries (hash 0, 1, ...) should have been evicted.
        remembered_hashes = {h for h, _ in cache._recent_hashes}
        assert "0000000000000000" not in remembered_hashes

    def test_store_seeds_the_fuzzy_window(self):
        cache, client = _make_cache()
        embedding = np.array([9.0], dtype=np.float32)
        cache.store("0000000000000000", embedding)

        client.get.return_value = None
        result = cache.lookup("0000000000000001")  # 1 bit away

        assert result.hit is True
        assert result.fuzzy is True
        np.testing.assert_array_equal(result.embedding, embedding)


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

    def test_flush_clears_the_fuzzy_window_too(self):
        cache, client = _make_cache()
        client.scan.return_value = (0, [])
        cache.store("0000000000000000", np.array([1.0], dtype=np.float32))
        assert len(cache._recent_hashes) == 1

        cache.flush()

        assert len(cache._recent_hashes) == 0
