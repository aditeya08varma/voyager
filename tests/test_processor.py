"""Tests for frame_processor module."""
import base64
import json
import time

import cv2
import numpy as np
import pytest

from processor.frame_processor import (
    decode_frame_message,
    _compute_content_hash,
    _compute_edge_density,
    FrameMetadata,
)


def _make_frame_message(frame: np.ndarray, camera_id="TEST-CAM", frame_id=0) -> bytes:
    _, jpeg = cv2.imencode(".jpg", frame)
    b64 = base64.b64encode(jpeg.tobytes()).decode("ascii")
    return json.dumps({
        "camera_id": camera_id,
        "frame_id": frame_id,
        "timestamp_ms": int(time.time() * 1000),
        "width": frame.shape[1],
        "height": frame.shape[0],
        "frame_b64": b64,
    }).encode()


class TestDecodeFrameMessage:
    def test_basic_decode(self):
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        raw = _make_frame_message(frame, "CAM-001", 42)
        meta, decoded = decode_frame_message(raw)
        assert meta.camera_id == "CAM-001"
        assert meta.frame_id == 42
        assert decoded.shape[0] == 100
        assert decoded.shape[1] == 100

    def test_metadata_fields(self):
        frame = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        raw = _make_frame_message(frame)
        meta, _ = decode_frame_message(raw)
        assert meta.width == 300
        assert meta.height == 200
        assert meta.channels == 3
        assert 0 <= meta.mean_brightness <= 255
        assert 0 <= meta.edge_density <= 1.0
        assert len(meta.content_hash) == 16

    def test_to_dict_roundtrip(self):
        frame = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        raw = _make_frame_message(frame)
        meta, _ = decode_frame_message(raw)
        d = meta.to_dict()
        assert isinstance(d, dict)
        assert "camera_id" in d
        assert "content_hash" in d


class TestContentHash:
    def test_same_frame_same_hash(self):
        frame = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        h1 = _compute_content_hash(frame)
        h2 = _compute_content_hash(frame)
        assert h1 == h2

    def test_different_frames_different_hash(self):
        frame_a = np.zeros((100, 100), dtype=np.uint8)
        frame_b = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(frame_b, (50, 50), 30, 255, -1)
        h1 = _compute_content_hash(frame_a)
        h2 = _compute_content_hash(frame_b)
        assert h1 != h2


class TestEdgeDensity:
    def test_blank_frame_low_density(self):
        frame = np.zeros((100, 100), dtype=np.uint8)
        density = _compute_edge_density(frame)
        assert density < 0.01

    def test_high_contrast_higher_density(self):
        frame = np.zeros((100, 100), dtype=np.uint8)
        frame[::2, :] = 255
        density = _compute_edge_density(frame)
        assert density > 0.01
