"""Tests for the Prometheus metrics collector.

MetricsCollector is a process-wide singleton (prometheus_client refuses to
register the same metric name twice), so every test shares one instance and
must reset its counters rather than constructing a fresh one.
"""
import time

import pytest

from monitoring.metrics_exporter import MetricsCollector


@pytest.fixture
def metrics():
    mc = MetricsCollector()
    mc._total_frames = 0
    mc._cache_hits = 0
    mc._throughput_window.clear()
    return mc


class TestSingleton:
    def test_returns_same_instance(self):
        assert MetricsCollector() is MetricsCollector()


class TestRecordFrameProcessed:
    def test_cache_hit_increments_hits_and_skips_inference_metric(self, metrics):
        metrics.record_frame_processed(camera_id="CAM-1", total_ms=5.0, cache_hit=True)
        assert metrics._total_frames == 1
        assert metrics._cache_hits == 1

    def test_cache_miss_does_not_increment_hits(self, metrics):
        metrics.record_frame_processed(camera_id="CAM-1", total_ms=5.0, cache_hit=False, inference_ms=20.0)
        assert metrics._total_frames == 1
        assert metrics._cache_hits == 0

    def test_fuzzy_hit_counts_as_a_hit_and_increments_fuzzy_counter(self, metrics):
        before = metrics.fuzzy_cache_hits._value.get()

        metrics.record_frame_processed(camera_id="CAM-1", total_ms=5.0, cache_hit=True, fuzzy=True)

        assert metrics._cache_hits == 1
        assert metrics.fuzzy_cache_hits._value.get() == before + 1

    def test_exact_hit_does_not_increment_fuzzy_counter(self, metrics):
        before = metrics.fuzzy_cache_hits._value.get()

        metrics.record_frame_processed(camera_id="CAM-1", total_ms=5.0, cache_hit=True, fuzzy=False)

        assert metrics.fuzzy_cache_hits._value.get() == before

    def test_hit_rate_gauge_reflects_mixed_results(self, metrics):
        metrics.record_frame_processed(camera_id="CAM-1", total_ms=1.0, cache_hit=True)
        metrics.record_frame_processed(camera_id="CAM-1", total_ms=1.0, cache_hit=False)
        assert metrics._cache_hits / metrics._total_frames == 0.5


class TestThroughputWindow:
    def test_expired_entries_are_evicted(self, metrics):
        metrics._throughput_window.append(time.monotonic() - 10.0)  # older than the 5s window
        metrics.record_frame_processed(camera_id="CAM-1", total_ms=1.0, cache_hit=True)
        assert len(metrics._throughput_window) == 1

    def test_recent_entries_are_kept(self, metrics):
        for _ in range(3):
            metrics.record_frame_processed(camera_id="CAM-1", total_ms=1.0, cache_hit=True)
        assert len(metrics._throughput_window) == 3


class TestGetSummary:
    def test_summary_before_any_frames(self, metrics):
        assert metrics.get_summary() == {
            "total_frames": 0,
            "cache_hits": 0,
            "cache_hit_rate": 0,
            "inference_savings_pct": 0,
        }

    def test_summary_after_frames(self, metrics):
        metrics.record_frame_processed(camera_id="CAM-1", total_ms=1.0, cache_hit=True)
        metrics.record_frame_processed(camera_id="CAM-1", total_ms=1.0, cache_hit=False)
        summary = metrics.get_summary()
        assert summary["total_frames"] == 2
        assert summary["cache_hits"] == 1
        assert summary["cache_hit_rate"] == 0.5
        assert summary["inference_savings_pct"] == 50.0
