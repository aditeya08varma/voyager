"""
Perceptual hashing algorithms for near-duplicate frame detection.

Provides dHash (difference hash), aHash (average hash), and pHash
(perceptual hash via DCT) with utilities for comparing similarity.
"""
from __future__ import annotations

import cv2
import numpy as np


def dhash(frame: np.ndarray, hash_size: int = 16) -> str:
    """Difference hash: compares adjacent pixel intensities."""
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(frame, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return "".join(str(int(b)) for b in diff.flatten())


def ahash(frame: np.ndarray, hash_size: int = 16) -> str:
    """Average hash: thresholds against mean pixel value."""
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(frame, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean_val = resized.mean()
    return "".join("1" if p > mean_val else "0" for p in resized.flatten())


def phash(frame: np.ndarray, hash_size: int = 16, highfreq_factor: int = 4) -> str:
    """Perceptual hash via DCT (discrete cosine transform)."""
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_size = hash_size * highfreq_factor
    resized = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    dct_low = dct[:hash_size, :hash_size]
    med = np.median(dct_low)
    return "".join("1" if v > med else "0" for v in dct_low.flatten())


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Number of differing bits between two hash strings."""
    if len(hash_a) != len(hash_b):
        raise ValueError(f"Hash length mismatch: {len(hash_a)} vs {len(hash_b)}")
    return sum(a != b for a, b in zip(hash_a, hash_b))


def are_similar(hash_a: str, hash_b: str, threshold: int = 10) -> bool:
    """Check if two hashes are perceptually similar (within threshold)."""
    return hamming_distance(hash_a, hash_b) <= threshold
