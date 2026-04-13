"""Tests for loadtest utilities."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loadtest.stress_test import (
    _generate_test_frame,
    _encode_frame_message,
    LoadTestResult,
)
from processor.frame_processor import decode_frame_message


class TestGenerateTestFrame:
    def test_frame_shape(self):
        frame = _generate_test_frame(0, 10)
        assert frame.shape == (480, 640, 3)

    def test_different_indices_different_content(self):
        f1 = _generate_test_frame(0, 100)
        f2 = _generate_test_frame(50, 100)
        assert not np.array_equal(f1, f2)


class TestEncodeDecodeRoundtrip:
    def test_roundtrip(self):
        frame = _generate_test_frame(5, 20)
        raw = _encode_frame_message(frame, "RT-TEST", 5)
        meta, decoded = decode_frame_message(raw)
        assert meta.camera_id == "RT-TEST"
        assert meta.frame_id == 5
        assert decoded.shape[0] == 480
        assert decoded.shape[1] == 640


class TestLoadTestResult:
    def test_creation(self):
        r = LoadTestResult(
            test_name="test",
            total_frames=100,
            duration_seconds=1.0,
            avg_latency_ms=5.0,
            p50_latency_ms=4.0,
            p95_latency_ms=10.0,
            p99_latency_ms=15.0,
            throughput_fps=100.0,
            errors=0,
            extra={},
        )
        assert r.test_name == "test"
        assert r.throughput_fps == 100.0
