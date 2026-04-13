"""
Kafka producer that simulates multiple concurrent camera feeds.

Each CameraSimulator runs in its own thread, reads frames from a video
file, encodes them as JPEG → Base64, and publishes to a Kafka topic.
The message key is the camera_id to ensure per-camera ordering.
"""
from __future__ import annotations

import base64
import json
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, asdict

import cv2
import numpy as np
import structlog
from confluent_kafka import Producer, KafkaError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import settings

log = structlog.get_logger(__name__)


@dataclass
class FrameMessage:
    camera_id: str
    frame_id: int
    timestamp_ms: int
    width: int
    height: int
    frame_b64: str


def create_producer() -> Producer:
    """Create a configured Kafka producer optimized for high throughput."""
    conf = {
        "bootstrap.servers": settings.kafka.bootstrap_servers,
        "message.max.bytes": settings.kafka.message_max_bytes,
        "linger.ms": 5,
        "batch.num.messages": 100,
        "compression.type": "lz4",
        "acks": "1",
        "queue.buffering.max.messages": 100000,
        "queue.buffering.max.kbytes": 1048576,
    }
    return Producer(conf)


def _delivery_callback(err, msg):
    if err:
        log.error("kafka_delivery_failed", error=str(err), topic=msg.topic())
    else:
        log.debug(
            "frame_delivered",
            camera=msg.key().decode() if msg.key() else "?",
            partition=msg.partition(),
            offset=msg.offset(),
        )


class CameraSimulator(threading.Thread):
    """Simulates a single camera feed by reading a video and publishing frames."""

    def __init__(
        self,
        camera_id: str,
        video_path: str,
        producer: Producer,
        topic: str,
        frame_interval_ms: int = 100,
    ):
        super().__init__(daemon=True, name=f"cam-{camera_id}")
        self.camera_id = camera_id
        self.video_path = video_path
        self.producer = producer
        self.topic = topic
        self.frame_interval_ms = frame_interval_ms
        self._stop_event = threading.Event()
        self._frame_count = 0

    def stop(self):
        self._stop_event.set()

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            log.error("video_open_failed", camera=self.camera_id, path=self.video_path)
            return

        log.info(
            "camera_started",
            camera_id=self.camera_id,
            video=self.video_path,
            interval_ms=self.frame_interval_ms,
        )

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                self._publish_frame(frame)
                self._frame_count += 1

                time.sleep(self.frame_interval_ms / 1000.0)
        finally:
            cap.release()
            self.producer.flush(timeout=5)
            log.info(
                "camera_stop",
                camera_id=self.camera_id,
                total_frames=self._frame_count,
            )

    def _publish_frame(self, frame: np.ndarray):
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64 = base64.b64encode(jpeg.tobytes()).decode("ascii")

        msg = FrameMessage(
            camera_id=self.camera_id,
            frame_id=self._frame_count,
            timestamp_ms=int(time.time() * 1000),
            width=frame.shape[1],
            height=frame.shape[0],
            frame_b64=b64,
        )

        try:
            self.producer.produce(
                topic=self.topic,
                key=self.camera_id.encode(),
                value=json.dumps(asdict(msg)).encode(),
                callback=_delivery_callback,
            )
            self.producer.poll(0)
        except BufferError:
            log.warning("producer_buffer_full", camera=self.camera_id)
            self.producer.poll(1)


def run_multi_camera_producer():
    """Entry point: start multiple camera simulators and wait."""
    producer = create_producer()

    video_path = settings.producer.video_source
    if not os.path.exists(video_path):
        log.error("video_not_found", path=video_path)
        log.info("generating_sample_video")
        from producer.sample_video_generator import generate_sample_video
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        generate_sample_video(video_path)

    cameras: list[CameraSimulator] = []
    for i in range(settings.producer.num_cameras):
        cam = CameraSimulator(
            camera_id=f"CAM-{i:03d}",
            video_path=video_path,
            producer=producer,
            topic=settings.kafka.topic_frames,
            frame_interval_ms=settings.producer.frame_interval_ms,
        )
        cameras.append(cam)

    shutdown = threading.Event()

    def handle_signal(sig, frame):
        log.info("shutdown_signal_received")
        shutdown.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    for cam in cameras:
        cam.start()
        log.info("camera_thread_started", camera_id=cam.camera_id)

    shutdown.wait()

    for cam in cameras:
        cam.stop()
    for cam in cameras:
        cam.join(timeout=5)

    producer.flush(timeout=10)
    log.info("producer_shutdown_complete")


if __name__ == "__main__":
    run_multi_camera_producer()
