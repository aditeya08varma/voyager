"""
Redis-backed embedding cache.

Implements a read-through cache: check Redis for a precomputed embedding
keyed by the frame's exact content hash. On a clean miss, falls back to a
small in-memory window of recently-seen hashes and looks for one within a
Hamming-distance threshold — a genuine near-duplicate match, not just an
exact one. On a real miss, the caller runs inference and writes the result
back via store().
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import redis
import structlog

from config.settings import settings
from inference.perceptual_hash import hamming_distance

log = structlog.get_logger(__name__)


@dataclass
class CacheResult:
    hit: bool
    embedding: Optional[np.ndarray]
    lookup_ms: float
    fuzzy: bool = False


class EmbeddingCache:
    PREFIX = "voyager:emb:"

    # How close two content_hash values must be (out of 64 bits) to be
    # treated as the same frame for caching purposes.
    FUZZY_HAMMING_THRESHOLD = 3
    # How many recently-seen (hash, embedding) pairs to keep as fuzzy-match
    # candidates. Bounded and in-memory, not Redis-backed: this is a
    # per-process window, scanned linearly on every clean miss, so it has to
    # stay small enough to keep that scan well under a millisecond.
    FUZZY_WINDOW_SIZE = 200

    def __init__(self, redis_client: redis.Redis | None = None):
        self._r = redis_client or redis.Redis(
            host=settings.redis.host,
            port=settings.redis.port,
            db=settings.redis.db,
            decode_responses=False,
            socket_connect_timeout=2,
            socket_timeout=1,
        )
        self._ttl = settings.redis.embedding_ttl
        self._recent_hashes: deque[tuple[str, np.ndarray]] = deque(maxlen=self.FUZZY_WINDOW_SIZE)

        self.total_lookups = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.fuzzy_hits = 0

    @property
    def hit_rate(self) -> float:
        if self.total_lookups == 0:
            return 0.0
        return self.cache_hits / self.total_lookups

    def lookup(self, content_hash: str) -> CacheResult:
        t0 = time.perf_counter()
        self.total_lookups += 1
        key = f"{self.PREFIX}{content_hash}"

        try:
            raw = self._r.get(key)
        except redis.RedisError as e:
            log.warning("redis_lookup_error", error=str(e), hash=content_hash)
            raw = None  # fall through to the in-memory fuzzy fallback below

        if raw is not None:
            elapsed = (time.perf_counter() - t0) * 1000
            embedding = np.frombuffer(raw, dtype=np.float32)
            self.cache_hits += 1
            self._remember(content_hash, embedding)
            log.debug("cache_hit", hash=content_hash, latency_ms=round(elapsed, 3))
            return CacheResult(hit=True, embedding=embedding, lookup_ms=elapsed)

        fuzzy_embedding = self._find_fuzzy_match(content_hash)
        elapsed = (time.perf_counter() - t0) * 1000

        if fuzzy_embedding is not None:
            self.cache_hits += 1
            self.fuzzy_hits += 1
            self._remember(content_hash, fuzzy_embedding)
            log.debug("cache_fuzzy_hit", hash=content_hash, latency_ms=round(elapsed, 3))
            return CacheResult(hit=True, embedding=fuzzy_embedding, lookup_ms=elapsed, fuzzy=True)

        self.cache_misses += 1
        log.debug("cache_miss", hash=content_hash, latency_ms=round(elapsed, 3))
        return CacheResult(hit=False, embedding=None, lookup_ms=elapsed)

    def _remember(self, content_hash: str, embedding: np.ndarray) -> None:
        self._recent_hashes.append((content_hash, embedding))

    def _find_fuzzy_match(self, content_hash: str) -> Optional[np.ndarray]:
        query_bits = self._to_bits(content_hash)
        best_embedding = None
        best_distance = self.FUZZY_HAMMING_THRESHOLD + 1

        for candidate_hash, candidate_embedding in self._recent_hashes:
            distance = hamming_distance(query_bits, self._to_bits(candidate_hash))
            if distance <= self.FUZZY_HAMMING_THRESHOLD and distance < best_distance:
                best_distance = distance
                best_embedding = candidate_embedding

        return best_embedding

    @staticmethod
    def _to_bits(content_hash: str) -> str:
        return format(int(content_hash, 16), "064b")

    def store(self, content_hash: str, embedding: np.ndarray) -> None:
        embedding = embedding.astype(np.float32)
        self._remember(content_hash, embedding)
        key = f"{self.PREFIX}{content_hash}"
        try:
            self._r.setex(key, self._ttl, embedding.tobytes())
        except redis.RedisError as e:
            log.warning("redis_store_error", error=str(e), hash=content_hash)

    def get_stats(self) -> dict:
        return {
            "total_lookups": self.total_lookups,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "fuzzy_hits": self.fuzzy_hits,
            "hit_rate": round(self.hit_rate, 4),
        }

    def flush(self) -> None:
        """Clear all cached embeddings (use in testing only)."""
        cursor = 0
        while True:
            cursor, keys = self._r.scan(cursor, match=f"{self.PREFIX}*", count=100)
            if keys:
                self._r.delete(*keys)
            if cursor == 0:
                break
        self._recent_hashes.clear()
