"""Tests for VoyagerStreamProcessor's per-frame processing logic.

EmbeddingCache, ModelHandler, and S3Handler are patched out
(VoyagerStreamProcessor constructs them itself, with no injection point) so
these tests exercise the hit/miss branching, result shape, and S3 archival
buffering without touching Redis, loading a real model, or hitting S3/LocalStack.
"""
import base64
import json
import time
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from processor.flink_job import VoyagerStreamProcessor


@dataclass
class _FakeCacheResult:
    hit: bool
    embedding: object = None
    lookup_ms: float = 0.1
    fuzzy: bool = False


def _make_frame_message() -> bytes:
    frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    _, jpeg = cv2.imencode(".jpg", frame)
    b64 = base64.b64encode(jpeg.tobytes()).decode("ascii")
    return json.dumps({
        "camera_id": "CAM-001",
        "frame_id": 0,
        "timestamp_ms": int(time.time() * 1000),
        "width": frame.shape[1],
        "height": frame.shape[0],
        "frame_b64": b64,
    }).encode()


@pytest.fixture
def processor():
    with patch("processor.flink_job.EmbeddingCache") as MockCache, \
         patch("processor.flink_job.ModelHandler") as MockModel, \
         patch("processor.flink_job.S3Handler") as MockStorage:
        cache = MockCache.return_value
        cache.get_stats.return_value = {"total_lookups": 1, "cache_hits": 0, "cache_misses": 1, "hit_rate": 0.0}
        cache.hit_rate = 0.0
        model = MockModel.return_value
        model.generate_embedding.return_value = np.zeros(1280, dtype=np.float32)
        storage = MockStorage.return_value
        yield VoyagerStreamProcessor(), cache, model, storage


class TestProcessFrameCacheHit:
    def test_hit_skips_inference_and_reports_hit_status(self, processor):
        proc, cache, model, _ = processor
        cache.lookup.return_value = _FakeCacheResult(hit=True, embedding=np.zeros(1280, dtype=np.float32))

        result = proc.process_frame(_make_frame_message())

        model.generate_embedding.assert_not_called()
        cache.store.assert_not_called()
        assert result["cache_status"] == "HIT"
        assert result["embedding_dim"] == 1280

    def test_fuzzy_hit_reports_distinct_status(self, processor):
        proc, cache, model, _ = processor
        cache.lookup.return_value = _FakeCacheResult(
            hit=True, embedding=np.zeros(1280, dtype=np.float32), fuzzy=True
        )

        result = proc.process_frame(_make_frame_message())

        model.generate_embedding.assert_not_called()
        assert result["cache_status"] == "FUZZY_HIT"
        assert result["latency"]["inference_ms"] == 0.0


class TestProcessFrameCacheMiss:
    def test_miss_runs_inference_and_stores_result(self, processor):
        proc, cache, model, _ = processor
        cache.lookup.return_value = _FakeCacheResult(hit=False)

        result = proc.process_frame(_make_frame_message())

        model.generate_embedding.assert_called_once()
        cache.store.assert_called_once()
        assert result["cache_status"] == "MISS"
        assert result["embedding_dim"] == 1280


class TestProcessFrameResultShape:
    def test_result_includes_metadata_and_latency_breakdown(self, processor):
        proc, cache, _, _ = processor
        cache.lookup.return_value = _FakeCacheResult(hit=True, embedding=np.zeros(1280, dtype=np.float32))

        result = proc.process_frame(_make_frame_message())

        assert result["camera_id"] == "CAM-001"
        assert set(result["latency"].keys()) == {"decode_ms", "cache_lookup_ms", "inference_ms", "total_ms"}


class TestS3ArchivalBuffering:
    def test_buffer_accumulates_without_uploading_before_batch_size(self, processor):
        proc, cache, _, storage = processor
        cache.lookup.return_value = _FakeCacheResult(hit=True, embedding=np.zeros(1280, dtype=np.float32))

        for _ in range(proc.EMBEDDINGS_BATCH_SIZE - 1):
            proc.process_frame(_make_frame_message())

        storage.upload_embeddings_batch.assert_not_called()
        assert len(proc._embedding_buffers["CAM-001"]) == proc.EMBEDDINGS_BATCH_SIZE - 1

    def test_buffer_flushes_and_clears_at_batch_size(self, processor):
        proc, cache, _, storage = processor
        cache.lookup.return_value = _FakeCacheResult(hit=True, embedding=np.zeros(1280, dtype=np.float32))

        for _ in range(proc.EMBEDDINGS_BATCH_SIZE):
            proc.process_frame(_make_frame_message())

        storage.upload_embeddings_batch.assert_called_once()
        camera_id, batch = storage.upload_embeddings_batch.call_args.args
        assert camera_id == "CAM-001"
        assert len(batch) == proc.EMBEDDINGS_BATCH_SIZE
        assert proc._embedding_buffers["CAM-001"] == []

    def test_upload_failure_is_logged_and_does_not_crash_processing(self, processor):
        proc, cache, _, storage = processor
        cache.lookup.return_value = _FakeCacheResult(hit=True, embedding=np.zeros(1280, dtype=np.float32))
        storage.upload_embeddings_batch.side_effect = RuntimeError("S3 unreachable")

        for _ in range(proc.EMBEDDINGS_BATCH_SIZE):
            proc.process_frame(_make_frame_message())  # must not raise

        assert proc._embedding_buffers["CAM-001"] == []


class TestS3HandlerUnavailable:
    def test_processing_continues_when_s3handler_fails_to_construct(self):
        with patch("processor.flink_job.EmbeddingCache") as MockCache, \
             patch("processor.flink_job.ModelHandler"), \
             patch("processor.flink_job.S3Handler", side_effect=RuntimeError("no LocalStack")):
            cache = MockCache.return_value
            cache.get_stats.return_value = {}
            cache.hit_rate = 0.0
            cache.lookup.return_value = _FakeCacheResult(hit=True, embedding=np.zeros(1280, dtype=np.float32))

            proc = VoyagerStreamProcessor()
            assert proc.storage is None

            proc.process_frame(_make_frame_message())  # must not raise
            assert proc._embedding_buffers == {}
