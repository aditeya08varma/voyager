"""
Load testing utilities for the Voyager pipeline.

Tests producer throughput, processing latency, and cache effectiveness
under various scenarios.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np
import structlog

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import settings
from processor.frame_processor import decode_frame_message
from processor.cache_handler import EmbeddingCache
from inference.model_handler import ModelHandler
from producer.kafka_producer import create_producer, CameraSimulator

log = structlog.get_logger(__name__)


@dataclass
class LoadTestResult:
    test_name: str
    total_frames: int
    duration_seconds: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_fps: float
    errors: int
    extra: dict


def _generate_test_frame(idx: int, total: int) -> np.ndarray:
    """Create a deterministic test frame for reproducible tests."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    progress = idx / max(total - 1, 1)

    for row in range(480):
        shade = int(50 + 150 * progress + 30 * (row / 480))
        frame[row, :] = (shade % 256, (shade + 50) % 256, (shade + 100) % 256)

    cx = int(320 + 200 * np.sin(progress * np.pi * 4))
    cy = int(240 + 150 * np.cos(progress * np.pi * 3))
    cv2.circle(frame, (cx, cy), 50, (0, 0, 255), -1)

    return frame


def _encode_frame_message(frame: np.ndarray, camera_id: str, frame_id: int) -> bytes:
    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    b64 = base64.b64encode(jpeg.tobytes()).decode("ascii")
    msg = {
        "camera_id": camera_id,
        "frame_id": frame_id,
        "timestamp_ms": int(time.time() * 1000),
        "width": frame.shape[1],
        "height": frame.shape[0],
        "frame_b64": b64,
    }
    return json.dumps(msg).encode()


def test_producer_throughput(
    num_cameras: int = 4,
    duration_seconds: int = 10,
) -> LoadTestResult:
    """Measure Kafka ingestion rate with multiple camera simulators."""
    producer = create_producer()
    video_path = settings.producer.video_source

    cameras: list[CameraSimulator] = []
    for i in range(num_cameras):
        cam = CameraSimulator(
            camera_id=f"LOADTEST-{i:03d}",
            video_path=video_path,
            producer=producer,
            topic=settings.kafka.topic_frames,
            frame_interval_ms=10,
        )
        cameras.append(cam)

    t_start = time.perf_counter()
    for cam in cameras:
        cam.start()

    time.sleep(duration_seconds)

    for cam in cameras:
        cam.stop()
    for cam in cameras:
        cam.join(timeout=5)

    producer.flush(timeout=10)
    elapsed = time.perf_counter() - t_start

    total_frames = sum(cam._frame_count for cam in cameras)
    fps = total_frames / elapsed if elapsed > 0 else 0

    log.info("producer_throughput_result",
             cameras=num_cameras, frames=total_frames, fps=round(fps, 2))

    return LoadTestResult(
        test_name="producer_throughput",
        total_frames=total_frames,
        duration_seconds=round(elapsed, 2),
        avg_latency_ms=0,
        p50_latency_ms=0,
        p95_latency_ms=0,
        p99_latency_ms=0,
        throughput_fps=round(fps, 2),
        errors=0,
        extra={"num_cameras": num_cameras},
    )


def test_processing_latency(num_frames: int = 200) -> LoadTestResult:
    """Measure decode, hash, and inference latency in-memory (no Kafka)."""
    model = ModelHandler()
    cache = EmbeddingCache()
    cache.flush()

    latencies = []
    errors = 0

    for i in range(num_frames):
        frame = _generate_test_frame(i, num_frames)
        raw = _encode_frame_message(frame, "LATENCY-TEST", i)

        try:
            t0 = time.perf_counter()
            meta, decoded = decode_frame_message(raw)

            cache_result = cache.lookup(meta.content_hash)
            if not cache_result.hit:
                emb = model.generate_embedding(decoded)
                cache.store(meta.content_hash, emb)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)
        except Exception as e:
            errors += 1
            log.error("processing_error", frame=i, error=str(e))

        if (i + 1) % 50 == 0:
            log.info("latency_test_progress", frame=i + 1, total=num_frames)

    latencies_arr = np.array(latencies)
    total_time = sum(latencies) / 1000

    result = LoadTestResult(
        test_name="processing_latency",
        total_frames=num_frames,
        duration_seconds=round(total_time, 2),
        avg_latency_ms=round(float(np.mean(latencies_arr)), 3),
        p50_latency_ms=round(float(np.percentile(latencies_arr, 50)), 3),
        p95_latency_ms=round(float(np.percentile(latencies_arr, 95)), 3),
        p99_latency_ms=round(float(np.percentile(latencies_arr, 99)), 3),
        throughput_fps=round(num_frames / total_time, 2) if total_time > 0 else 0,
        errors=errors,
        extra={"cache_stats": cache.get_stats()},
    )

    log.info("latency_test_complete", **{
        "avg_ms": result.avg_latency_ms,
        "p95_ms": result.p95_latency_ms,
        "cache_hit_rate": cache.hit_rate,
    })
    return result


def test_cache_hit_simulation(
    num_unique: int = 50,
    num_repeated: int = 150,
) -> LoadTestResult:
    """Demonstrate cache effectiveness with controlled unique vs recurring frames."""
    model = ModelHandler()
    cache = EmbeddingCache()
    cache.flush()

    unique_frames = [_generate_test_frame(i, num_unique) for i in range(num_unique)]

    latencies = []
    cache_hits = 0
    total = num_unique + num_repeated

    for i in range(num_unique):
        raw = _encode_frame_message(unique_frames[i], "CACHE-TEST", i)
        t0 = time.perf_counter()
        meta, decoded = decode_frame_message(raw)
        cache_result = cache.lookup(meta.content_hash)
        if not cache_result.hit:
            emb = model.generate_embedding(decoded)
            cache.store(meta.content_hash, emb)
        else:
            cache_hits += 1
        latencies.append((time.perf_counter() - t0) * 1000)

    for i in range(num_repeated):
        frame = unique_frames[i % num_unique]
        raw = _encode_frame_message(frame, "CACHE-TEST", num_unique + i)
        t0 = time.perf_counter()
        meta, decoded = decode_frame_message(raw)
        cache_result = cache.lookup(meta.content_hash)
        if not cache_result.hit:
            emb = model.generate_embedding(decoded)
            cache.store(meta.content_hash, emb)
        else:
            cache_hits += 1
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies_arr = np.array(latencies)
    total_time = sum(latencies) / 1000
    hit_rate = cache_hits / total

    result = LoadTestResult(
        test_name="cache_hit_simulation",
        total_frames=total,
        duration_seconds=round(total_time, 2),
        avg_latency_ms=round(float(np.mean(latencies_arr)), 3),
        p50_latency_ms=round(float(np.percentile(latencies_arr, 50)), 3),
        p95_latency_ms=round(float(np.percentile(latencies_arr, 95)), 3),
        p99_latency_ms=round(float(np.percentile(latencies_arr, 99)), 3),
        throughput_fps=round(total / total_time, 2) if total_time > 0 else 0,
        errors=0,
        extra={
            "cache_hits": cache_hits,
            "cache_hit_rate": round(hit_rate, 4),
            "inference_savings_pct": round(hit_rate * 100, 2),
            "unique_frames": num_unique,
            "repeated_frames": num_repeated,
        },
    )

    log.info("cache_simulation_complete", **result.extra)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Voyager Load Tests")
    parser.add_argument("--test", choices=["throughput", "latency", "cache", "all"],
                        default="all")
    args = parser.parse_args()

    results = []
    if args.test in ("throughput", "all"):
        results.append(test_producer_throughput())
    if args.test in ("latency", "all"):
        results.append(test_processing_latency())
    if args.test in ("cache", "all"):
        results.append(test_cache_hit_simulation())

    print("\n" + "=" * 60)
    for r in results:
        print(f"\n{r.test_name}:")
        print(f"  Frames:     {r.total_frames}")
        print(f"  Duration:   {r.duration_seconds}s")
        print(f"  Throughput: {r.throughput_fps} fps")
        print(f"  Avg:        {r.avg_latency_ms}ms")
        print(f"  P95:        {r.p95_latency_ms}ms")
        print(f"  P99:        {r.p99_latency_ms}ms")
        if r.extra:
            for k, v in r.extra.items():
                print(f"  {k}: {v}")
