"""
Prometheus metrics exporter for Voyager pipeline.

Exposes:
  - voyager_frames_processed_total       (Counter)
  - voyager_frame_processing_seconds     (Histogram)
  - voyager_cache_lookups_total          (Counter, label: status=hit|miss)
  - voyager_inference_duration_seconds   (Histogram)
  - voyager_cache_hit_rate               (Gauge)
  - voyager_inferences_skipped_total     (Counter)
  - voyager_throughput_fps               (Gauge)
"""
from __future__ import annotations

import threading

from prometheus_client import (
    Counter, Gauge, Histogram, Info, start_http_server,
    CollectorRegistry, REGISTRY,
)
import structlog

log = structlog.get_logger(__name__)

FRAME_LATENCY_BUCKETS = (
    0.005, 0.010, 0.020, 0.030, 0.040, 0.050,
    0.075, 0.100, 0.200, 0.500, 1.0, 2.0, 5.0,
)

INFERENCE_LATENCY_BUCKETS = (
    0.010, 0.025, 0.050, 0.100, 0.250, 0.500, 1.0, 2.0, 5.0,
)


class MetricsCollector:
    """Thread-safe Prometheus metrics collector for the Voyager pipeline."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.pipeline_info = Info(
            "voyager_pipeline",
            "Voyager video processing pipeline metadata",
        )
        self.pipeline_info.info({
            "version": "1.0.0",
            "model": "mobilenetv2",
        })

        self.frames_processed = Counter(
            "voyager_frames_processed_total",
            "Total video frames processed",
            ["camera_id"],
        )

        self.frame_latency = Histogram(
            "voyager_frame_processing_seconds",
            "End-to-end frame processing latency",
            ["camera_id"],
            buckets=FRAME_LATENCY_BUCKETS,
        )

        self.cache_lookups = Counter(
            "voyager_cache_lookups_total",
            "Total Redis cache lookups",
            ["status"],
        )

        self.inference_duration = Histogram(
            "voyager_inference_duration_seconds",
            "AI model inference duration",
            buckets=INFERENCE_LATENCY_BUCKETS,
        )

        self.cache_hit_rate = Gauge(
            "voyager_cache_hit_rate",
            "Current cache hit rate (0-1)",
        )

        self.inferences_skipped = Counter(
            "voyager_inferences_skipped_total",
            "Total inference calls avoided via cache hits",
        )

        self.throughput = Gauge(
            "voyager_throughput_fps",
            "Current processing throughput in frames/second",
        )

        self._total_frames = 0
        self._cache_hits = 0
        self._throughput_window: list[float] = []

    def record_frame_processed(
        self,
        camera_id: str,
        total_ms: float,
        cache_hit: bool,
        inference_ms: float = 0.0,
    ) -> None:
        import time

        self.frames_processed.labels(camera_id=camera_id).inc()
        self.frame_latency.labels(camera_id=camera_id).observe(total_ms / 1000.0)

        if cache_hit:
            self.cache_lookups.labels(status="hit").inc()
            self.inferences_skipped.inc()
            self._cache_hits += 1
        else:
            self.cache_lookups.labels(status="miss").inc()
            self.inference_duration.observe(inference_ms / 1000.0)

        self._total_frames += 1
        if self._total_frames > 0:
            self.cache_hit_rate.set(self._cache_hits / self._total_frames)

        now = time.monotonic()
        self._throughput_window.append(now)
        cutoff = now - 5.0
        self._throughput_window = [t for t in self._throughput_window if t > cutoff]
        fps = len(self._throughput_window) / 5.0
        self.throughput.set(round(fps, 2))

    def set_throughput(self, fps: float) -> None:
        self.throughput.set(fps)

    def get_summary(self) -> dict:
        rate = self._cache_hits / self._total_frames if self._total_frames > 0 else 0
        return {
            "total_frames": self._total_frames,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": round(rate, 4),
            "inference_savings_pct": round(rate * 100, 2),
        }


def start_metrics_server(port: int = 8000) -> None:
    """Start the Prometheus HTTP metrics endpoint."""
    start_http_server(port)
    log.info("metrics_server_started", port=port, endpoint=f"http://localhost:{port}/metrics")
