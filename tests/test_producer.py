"""Tests for producer module."""
import os
import json
import time
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from producer.kafka_producer import FrameMessage, CameraSimulator, create_producer
from producer.sample_video_generator import generate_sample_video


class TestFrameMessage:
    def test_creation(self):
        msg = FrameMessage(
            camera_id="CAM-001",
            frame_id=0,
            timestamp_ms=int(time.time() * 1000),
            width=640,
            height=480,
            frame_b64="dGVzdA==",
        )
        assert msg.camera_id == "CAM-001"
        assert msg.width == 640


class TestSampleVideoGenerator:
    def test_generates_file(self, tmp_path):
        out = str(tmp_path / "test.mp4")
        result = generate_sample_video(out, duration_seconds=2, fps=5)
        assert os.path.exists(result)
        cap = cv2.VideoCapture(result)
        assert cap.isOpened()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert frame_count >= 8
        cap.release()

    def test_frame_dimensions(self, tmp_path):
        out = str(tmp_path / "test.mp4")
        generate_sample_video(out, duration_seconds=1, fps=5, width=320, height=240)
        cap = cv2.VideoCapture(out)
        ret, frame = cap.read()
        assert ret
        assert frame.shape[1] == 320
        assert frame.shape[0] == 240
        cap.release()


class TestCameraSimulator:
    def test_stop_event(self):
        producer = MagicMock()
        cam = CameraSimulator(
            camera_id="TEST",
            video_path="nonexistent.mp4",
            producer=producer,
            topic="test-topic",
        )
        cam.stop()
        assert cam._stop_event.is_set()
