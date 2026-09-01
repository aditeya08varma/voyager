"""
Frame decoding and metadata extraction utilities.

Operates on raw Kafka messages to produce structured frame metadata
suitable for downstream inference and caching.
"""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import orjson


@dataclass(slots=True)
class FrameMetadata:
    camera_id: str
    frame_id: int
    timestamp_ms: int
    width: int
    height: int
    channels: int
    content_hash: str
    mean_brightness: float
    edge_density: float
    decode_latency_ms: float
    processing_ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "frame_id": self.frame_id,
            "timestamp_ms": self.timestamp_ms,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "content_hash": self.content_hash,
            "mean_brightness": round(self.mean_brightness, 2),
            "edge_density": round(self.edge_density, 4),
            "decode_latency_ms": round(self.decode_latency_ms, 3),
            "processing_ts": self.processing_ts,
        }

    def to_json(self) -> bytes:
        return orjson.dumps(self.to_dict())


def decode_frame_message(raw_value: bytes) -> tuple[FrameMetadata, np.ndarray]:
    """Decode a Kafka message into metadata and a numpy frame array.

    Returns (metadata, frame_array) where frame_array is the decoded BGR image.
    """
    t0 = time.perf_counter()
    msg = orjson.loads(raw_value)

    frame_bytes = base64.b64decode(msg["frame_b64"])
    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError(f"Failed to decode frame {msg.get('frame_id')} from {msg.get('camera_id')}")

    decode_ms = (time.perf_counter() - t0) * 1000

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    content_hash = _compute_content_hash(gray)
    brightness = float(np.mean(gray))
    edge_density = _compute_edge_density(gray)

    meta = FrameMetadata(
        camera_id=msg["camera_id"],
        frame_id=msg["frame_id"],
        timestamp_ms=msg["timestamp_ms"],
        width=frame.shape[1],
        height=frame.shape[0],
        channels=frame.shape[2],
        content_hash=content_hash,
        mean_brightness=brightness,
        edge_density=edge_density,
        decode_latency_ms=decode_ms,
    )
    return meta, frame


def _compute_content_hash(gray: np.ndarray) -> str:
    """Fast 8x8 DCT-based hash for near-duplicate detection.

    Downscales to 8x8, computes mean, and produces a 64-bit binary hash.
    Identical to the classic dHash approach but simpler.
    """
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    hash_int = sum(2**i for i, v in enumerate(diff.flatten()) if v)
    return f"{hash_int:016x}"


def _compute_edge_density(gray: np.ndarray) -> float:
    """Fraction of pixels that are edges (Canny), used as a scene-activity signal."""
    edges = cv2.Canny(gray, 50, 150)
    return float(np.count_nonzero(edges)) / edges.size
