"""Video loading and frame sampling utilities."""
from __future__ import annotations
import logging
from typing import List, Tuple

import cv2
import numpy as np

from .types import VideoLoadError

logger = logging.getLogger(__name__)


def load_video(path: str) -> List[np.ndarray]:
    """Load all frames from a video file.

    Args:
        path: Path to the video file.

    Returns:
        List of frames as numpy arrays (H x W x C, BGR).

    Raises:
        VideoLoadError: If the file cannot be opened or has fewer than 25 frames.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoLoadError(f"Cannot open video file: {path}")

    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) < 25:
        raise VideoLoadError(
            f"Video '{path}' has only {len(frames)} frames (minimum 25 required)."
        )
    return frames


def sample_frames(
    video_path: str,
    n: int = 25,
    size: Tuple[int, int] = (448, 448),
) -> List[np.ndarray]:
    """Uniformly sample n frames from a video and resize each to `size`.

    Args:
        video_path: Path to the video file.
        n: Number of frames to sample (default 25).
        size: Target (width, height) for each frame (default 448x448).

    Returns:
        List of n frames as numpy arrays (H x W x C, RGB uint8).

    Raises:
        VideoLoadError: If the video cannot be loaded.
    """
    frames = load_video(video_path)
    total = len(frames)
    indices = [int(i * (total - 1) / (n - 1)) for i in range(n)]
    sampled = []
    for idx in indices:
        frame = frames[idx]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, size, interpolation=cv2.INTER_LINEAR)
        sampled.append(resized)
    return sampled


def frames_from_array(
    frames: List[np.ndarray],
    n: int = 25,
    size: Tuple[int, int] = (448, 448),
) -> List[np.ndarray]:
    """Uniformly sample n frames from an in-memory frame list and resize.

    Args:
        frames: Pre-loaded list of frames (H x W x C, BGR or RGB).
        n: Number of frames to sample.
        size: Target (width, height).

    Returns:
        List of n resized frames (RGB uint8).

    Raises:
        VideoLoadError: If fewer than 25 frames are provided.
    """
    if len(frames) < 25:
        raise VideoLoadError(
            f"Frame list has only {len(frames)} frames (minimum 25 required)."
        )
    total = len(frames)
    indices = [int(i * (total - 1) / (n - 1)) for i in range(n)]
    sampled = []
    for idx in indices:
        frame = frames[idx]
        resized = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
        sampled.append(resized)
    return sampled
