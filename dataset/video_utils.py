"""Video loading and frame sampling utilities."""
from __future__ import annotations
import logging
from typing import List, Tuple

import cv2
import numpy as np

from .types import VideoLoadError

logger = logging.getLogger(__name__)


def uniform_sample_indices(total_frames: int, n: int = 25) -> List[int]:
    """Return the exact integer indices used by the uniform frame sampler."""
    if n < 2:
        raise ValueError("n must be at least 2.")
    if total_frames < n:
        raise VideoLoadError(
            f"Video has {total_frames} frames but {n} samples are required."
        )
    return [
        int(index * (total_frames - 1) / (n - 1)) for index in range(n)
    ]


def uniform_sample_timestamps(
    total_frames: int, fps: float, n: int = 25
) -> List[float]:
    """Map the exact sampled frame indices to timestamps in seconds."""
    if fps <= 0.0:
        raise ValueError("fps must be positive.")
    return [
        frame_index / fps
        for frame_index in uniform_sample_indices(total_frames, n)
    ]


def probe_video(path: str) -> Tuple[float, int, float]:
    """Return `(fps, frame_count, duration_sec)` from the source container."""
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise VideoLoadError(f"Cannot open video file: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0.0 or frame_count <= 1:
        raise VideoLoadError(
            f"Invalid video metadata for '{path}': fps={fps}, frames={frame_count}."
        )
    return fps, frame_count, frame_count / fps


def load_video(path: str) -> List[np.ndarray]:
    """Load all frames from a video file.

    Args:
        path: Path to the video file.

    Returns:
        List of frames as numpy arrays (H x W x C, RGB).

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
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if len(frames) < 25:
        raise VideoLoadError(
            f"Video '{path}' has only {len(frames)} frames (minimum 25 required)."
        )
    return frames


def sample_frames(
    video_path: str,
    n: int = 25,
    size: Tuple[int, int] = (224, 224),
) -> List[np.ndarray]:
    """Uniformly sample n frames from a video and resize each to `size`.

    Args:
        video_path: Path to the video file.
        n: Number of frames to sample (default 25).
        size: Target (width, height) for each frame (default 224x224).

    Returns:
        List of n frames as numpy arrays (H x W x C, RGB uint8).

    Raises:
        VideoLoadError: If the video cannot be loaded.
    """
    frames = load_video(video_path)
    total = len(frames)
    indices = uniform_sample_indices(total, n)
    sampled = []
    for idx in indices:
        frame = frames[idx]
        resized = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
        sampled.append(resized)
    return sampled


def frames_from_array(
    frames: List[np.ndarray],
    n: int = 25,
    size: Tuple[int, int] = (224, 224),
) -> List[np.ndarray]:
    """Uniformly sample n frames from an in-memory frame list and resize.

    Args:
        frames: Pre-loaded list of RGB frames (H x W x C).
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
    indices = uniform_sample_indices(total, n)
    sampled = []
    for idx in indices:
        frame = frames[idx]
        resized = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
        sampled.append(resized)
    return sampled
