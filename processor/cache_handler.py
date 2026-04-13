"""
Redis-backed embedding cache.

Implements a read-through cache: check Redis for a precomputed embedding
keyed by the frame's perceptual hash. On miss, the caller runs inference
and writes the result back.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import redis
import structlog

from config.settings import settings

log = structlog.get_logger(__name__)


@dataclass
class CacheResult:
    hit: bool
    embedding: Optional[np.ndarray]
    lookup_ms: float


class EmbeddingCache:
    PREFIX = "voyager:emb:"

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

        self.total_lookups = 0
        self.cache_hits = 0
        self.cache_misses = 0

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
            self.cache_misses += 1
            return CacheResult(hit=False, embedding=None,
                               lookup_ms=(time.perf_counter() - t0) * 1000)

        elapsed = (time.perf_counter() - t0) * 1000

        if raw is not None:
            self.cache_hits += 1
            embedding = np.frombuffer(raw, dtype=np.float32)
            log.debug("cache_hit", hash=content_hash, latency_ms=round(elapsed, 3))
            return CacheResult(hit=True, embedding=embedding, lookup_ms=elapsed)

        self.cache_misses += 1
        log.debug("cache_miss", hash=content_hash, latency_ms=round(elapsed, 3))
        return CacheResult(hit=False, embedding=None, lookup_ms=elapsed)

    def store(self, content_hash: str, embedding: np.ndarray) -> None:
        key = f"{self.PREFIX}{content_hash}"
        try:
            self._r.setex(key, self._ttl, embedding.astype(np.float32).tobytes())
        except redis.RedisError as e:
            log.warning("redis_store_error", error=str(e), hash=content_hash)

    def get_stats(self) -> dict:
        return {
            "total_lookups": self.total_lookups,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
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
