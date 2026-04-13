"""
Generates synthetic sample video data for testing the Voyager pipeline.

Creates an MP4 with a mix of dynamic frames (moving shapes) and
recurring static backgrounds — the latter are designed to produce
cache hits when processed through the perceptual hashing + Redis cache.
"""
from __future__ import annotations

import os

import cv2
import numpy as np


def generate_sample_video(
    output_path: str = "./data/sample_videos/sample.mp4",
    duration_seconds: int = 30,
    fps: int = 15,
    width: int = 640,
    height: int = 480,
    num_recurring_bg_frames: int = 5,
) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_seconds * fps
    backgrounds = _generate_backgrounds(num_recurring_bg_frames, width, height)

    for i in range(total_frames):
        if (i % 20) < 3:
            frame = backgrounds[i % len(backgrounds)].copy()
        else:
            frame = _render_dynamic_frame(i, total_frames, width, height)

        ts_text = f"F:{i:05d} T:{i / fps:.2f}s"
        cv2.putText(frame, ts_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"CAM-SIM", (width - 140, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        writer.write(frame)

    writer.release()
    return output_path


def _generate_backgrounds(count: int, w: int, h: int) -> list[np.ndarray]:
    backgrounds = []
    for i in range(count):
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        shade = 30 + i * 10
        bg[:] = (shade, shade + 5, shade + 10)
        cv2.rectangle(bg, (50, 50), (w - 50, h - 50), (shade + 40, shade + 30, shade + 20), 3)
        backgrounds.append(bg)
    return backgrounds


def _render_dynamic_frame(idx: int, total: int, w: int, h: int) -> np.ndarray:
    progress = idx / max(total - 1, 1)

    frame = np.zeros((h, w, 3), dtype=np.uint8)
    for row in range(h):
        r = int(30 + 50 * progress + 20 * (row / h))
        g = int(50 + 30 * (1 - progress) + 10 * (row / h))
        b = int(70 + 40 * progress - 20 * (row / h))
        frame[row, :] = (max(0, min(255, b)), max(0, min(255, g)), max(0, min(255, r)))

    cx = int(w * (0.2 + 0.6 * progress))
    cy = int(h * (0.3 + 0.4 * np.sin(progress * np.pi * 4)))
    cv2.circle(frame, (cx, cy), 40, (0, 0, 255), -1)
    cv2.circle(frame, (cx, cy), 42, (255, 255, 255), 2)

    rx = int(w * (0.8 - 0.6 * progress))
    ry = int(h * (0.7 - 0.4 * np.cos(progress * np.pi * 3)))
    cv2.rectangle(frame, (rx - 30, ry - 30), (rx + 30, ry + 30), (255, 200, 0), -1)

    tx = int(w * 0.5 + 100 * np.sin(progress * np.pi * 6))
    ty = int(h * 0.5 + 80 * np.cos(progress * np.pi * 5))
    pts = np.array([
        [tx, ty - 25],
        [tx - 25, ty + 20],
        [tx + 25, ty + 20],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [pts], (0, 255, 100))

    return frame


if __name__ == "__main__":
    path = generate_sample_video()
    print(f"Generated sample video: {path}")
