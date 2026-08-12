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


def native_motion_pairs(total_frames: int, n: int = 25, offset: int = 1):
    """Return anchor/adjacent native-frame pairs and never jump anchor-to-anchor."""
    if offset < 1:
        raise ValueError("offset must be at least one native frame.")
    if total_frames < n + offset:
        raise VideoLoadError(
            f"Video has {total_frames} frames but {n} anchors with offset "
            f"{offset} require at least {n + offset}."
        )
    anchors = [
        int(index * (total_frames - 1 - offset) / (n - 1))
        for index in range(n)
    ]
    return [(anchor, anchor + offset) for anchor in anchors]


def farneback_native_flow(
    frames: List[np.ndarray],
    n: int = 25,
    offset: int = 1,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """Reference frozen flow estimator over adjacent native-rate frame pairs.

    Farnebäck has no trainable parameters and makes the release runnable
    without redistributing a third-party neural checkpoint.  A full experiment
    may replace it with another frozen estimator, but that estimator and its
    checkpoint hash must be recorded in provenance.
    """
    pairs = native_motion_pairs(len(frames), n=n, offset=offset)
    flows = []
    for first_index, second_index in pairs:
        first = cv2.cvtColor(frames[first_index], cv2.COLOR_RGB2GRAY)
        second = cv2.cvtColor(frames[second_index], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            first,
            second,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        flows.append(flow.astype(np.float32))
    return flows, pairs


def estimate_training_velocity_scale(
    videos: List[List[np.ndarray]],
    fps_values: List[float],
    *,
    n: int = 25,
    quantile: float = 0.99,
) -> float:
    """Estimate the fixed ``v_max`` from training-source native-rate flow."""
    if len(videos) != len(fps_values) or not videos:
        raise ValueError("videos and fps_values must be non-empty aligned lists.")
    if not 0.5 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0.5, 1.0].")
    magnitudes = []
    for frames, fps in zip(videos, fps_values):
        if fps <= 0.0:
            raise ValueError("fps values must be positive.")
        flows, pairs = farneback_native_flow(frames, n=n)
        for flow, (first_index, second_index) in zip(flows, pairs):
            elapsed = (second_index - first_index) / float(fps)
            velocity = flow / elapsed
            magnitudes.append(np.linalg.norm(velocity, axis=-1).reshape(-1))
    value = float(np.quantile(np.concatenate(magnitudes), quantile))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("Training flow did not produce a positive v_max.")
    return value


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


def sample_anchor_motion_frames(
    video_path: str,
    n: int = 25,
    size: Tuple[int, int] = (224, 224),
    offset: int = 1,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[int, int]]]:
    """Load anchor frames and their adjacent native-rate motion partners."""
    frames = load_video(video_path)
    pairs = native_motion_pairs(len(frames), n=n, offset=offset)
    anchors = [
        cv2.resize(frames[first], size, interpolation=cv2.INTER_LINEAR)
        for first, _ in pairs
    ]
    partners = [
        cv2.resize(frames[second], size, interpolation=cv2.INTER_LINEAR)
        for _, second in pairs
    ]
    return anchors, partners, pairs


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
