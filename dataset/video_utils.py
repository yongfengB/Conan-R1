"""Video loading and frame sampling utilities."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
import tempfile
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .types import VideoLoadError

logger = logging.getLogger(__name__)

FARNEBACK_PARAMETERS = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 15,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
}
MOTION_FRAME_SIZE = (224, 224)
MOTION_ANCHORS = 25
MOTION_NATIVE_OFFSET = 1
MOTION_SCALE_SAMPLING_METHOD = "source_keyed_uniform_without_replacement_v1"


def motion_scale_contract(method: dict) -> dict[str, Any]:
    """Resolve the one preprocessing/statistics contract used by code and JSON."""
    motion = method["motion"]
    normalization = motion["normalization"]
    estimation = normalization["estimation"]
    if estimation.get("sampling_method") != MOTION_SCALE_SAMPLING_METHOD:
        raise ValueError("method_config uses an unsupported motion-scale sampler.")
    frame_size = int(method["appearance_encoder"]["frame_size"])
    return {
        "estimator": str(motion["estimator"]),
        "quantile": float(normalization["quantile"]),
        "motion_preprocessing": {
            "anchors": int(method["appearance_encoder"]["anchors"]),
            "frame_size": [frame_size, frame_size],
            "native_frame_offset": int(motion["native_frame_offset"]),
            "resize_interpolation": "opencv_inter_linear",
            "flow_parameters": validate_farneback_parameters(
                motion["flow_parameters"]
            ),
        },
        "sampling": {
            "method": MOTION_SCALE_SAMPLING_METHOD,
            "seed": int(estimation["seed"]),
            "samples_per_source": int(estimation["samples_per_source"]),
            "storage": "temporary_disk_memmap",
        },
    }


def validate_motion_scale_payload(
    payload: dict,
    method: dict,
    *,
    method_config_sha256: str,
    num_frames: int,
    frame_size: int,
) -> float:
    """Reject a scale estimated under any preprocessing other than training."""
    contract = motion_scale_contract(method)
    expected_preprocessing = contract["motion_preprocessing"]
    if int(num_frames) != expected_preprocessing["anchors"]:
        raise ValueError("Training num_frames must match the motion-scale anchors.")
    if int(frame_size) != expected_preprocessing["frame_size"][0]:
        raise ValueError("Training frame_size must match the motion-scale frame_size.")
    mismatches = {}
    expected_top_level = {
        "schema_version": 2,
        "estimator": contract["estimator"],
        "quantile": contract["quantile"],
        "motion_preprocessing": expected_preprocessing,
        "method_config_sha256": method_config_sha256,
    }
    for key, value in expected_top_level.items():
        if payload.get(key) != value:
            mismatches[key] = payload.get(key)
    sampling = payload.get("sampling", {})
    if any(
        sampling.get(key) != value
        for key, value in contract["sampling"].items()
    ) or int(sampling.get("sampled_values", 0)) < 1:
        mismatches["sampling"] = sampling
    if mismatches:
        raise ValueError(
            "motion_scale.json does not match the training motion protocol: "
            f"{mismatches}"
        )
    value = float(payload.get("v_max", 0.0))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("motion_scale.json v_max must be finite and positive.")
    return value


def validate_farneback_parameters(parameters: dict) -> dict:
    """Return a normalized complete parameter mapping for the frozen flow."""
    if set(parameters) != set(FARNEBACK_PARAMETERS):
        raise ValueError(
            "Farneback parameters must define exactly "
            f"{sorted(FARNEBACK_PARAMETERS)}."
        )
    normalized = {
        "pyr_scale": float(parameters["pyr_scale"]),
        "levels": int(parameters["levels"]),
        "winsize": int(parameters["winsize"]),
        "iterations": int(parameters["iterations"]),
        "poly_n": int(parameters["poly_n"]),
        "poly_sigma": float(parameters["poly_sigma"]),
        "flags": int(parameters["flags"]),
    }
    if not 0.0 < normalized["pyr_scale"] < 1.0:
        raise ValueError("Farneback pyr_scale must lie in (0, 1).")
    if min(
        normalized["levels"],
        normalized["winsize"],
        normalized["iterations"],
        normalized["poly_n"],
    ) < 1:
        raise ValueError("Farneback integer parameters must be positive.")
    if normalized["poly_sigma"] <= 0.0:
        raise ValueError("Farneback poly_sigma must be positive.")
    return normalized


def farneback_pair_flow(
    first: np.ndarray,
    second: np.ndarray,
    parameters: dict = FARNEBACK_PARAMETERS,
) -> np.ndarray:
    """Estimate one RGB-frame displacement field with explicit parameters."""
    parameters = validate_farneback_parameters(parameters)
    if first.shape[:2] != second.shape[:2]:
        second = cv2.resize(
            second, (first.shape[1], first.shape[0]), interpolation=cv2.INTER_LINEAR
        )
    first_gray = cv2.cvtColor(first.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    second_gray = cv2.cvtColor(second.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(
        first_gray, second_gray, None, **parameters
    ).astype(np.float32)


def dense_farneback_flows(
    anchors: Sequence[np.ndarray],
    partners: Sequence[np.ndarray],
    parameters: dict = FARNEBACK_PARAMETERS,
) -> np.ndarray:
    """Compute the exact dense-flow tensor shared by training and scale fitting."""
    if len(anchors) != len(partners) or not anchors:
        raise ValueError("Each anchor needs one adjacent native-rate motion frame.")
    normalized = validate_farneback_parameters(parameters)
    return np.stack(
        [
            farneback_pair_flow(first, second, normalized)
            for first, second in zip(anchors, partners)
        ]
    ).astype(np.float32)


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


@dataclass(frozen=True)
class MotionScaleEstimate:
    v_max: float
    sampled_values: int
    source_count: int
    samples_per_source: int
    sampling_seed: int


def _source_sampling_seed(source_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{source_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def estimate_training_velocity_scale(
    sources: Sequence[Tuple[str, str, float]],
    *,
    n: int = MOTION_ANCHORS,
    size: Tuple[int, int] = MOTION_FRAME_SIZE,
    offset: int = MOTION_NATIVE_OFFSET,
    quantile: float = 0.99,
    parameters: dict = FARNEBACK_PARAMETERS,
    samples_per_source: int = 4096,
    sampling_seed: int = 42,
    work_dir: Optional[str] = None,
) -> MotionScaleEstimate:
    """Fit ``v_max`` with the training motion path and bounded disk storage.

    Each source contributes a deterministic, source-keyed sample from its
    224x224 native-adjacent flow field. Samples are written to a disk-backed
    array, so neither decoded videos nor full-split pixel flows accumulate in
    memory.
    """
    if not sources:
        raise ValueError("At least one training source is required.")
    if not 0.5 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0.5, 1.0].")
    if samples_per_source < 1:
        raise ValueError("samples_per_source must be positive.")
    if len({source_id for source_id, _, _ in sources}) != len(sources):
        raise ValueError("Training motion-scale sources must have unique ids.")
    parameters = validate_farneback_parameters(parameters)
    maximum_samples = len(sources) * samples_per_source
    with tempfile.TemporaryDirectory(dir=work_dir) as temporary_directory:
        sample_path = Path(temporary_directory) / "velocity_samples.float32"
        samples = np.memmap(
            sample_path, dtype=np.float32, mode="w+", shape=(maximum_samples,)
        )
        cursor = 0
        for source_id, video_path, fps in sorted(sources, key=lambda item: item[0]):
            if fps <= 0.0:
                raise ValueError("fps values must be positive.")
            probed_fps, _, _ = probe_video(video_path)
            if not np.isclose(probed_fps, fps, rtol=1e-4, atol=1e-4):
                raise ValueError(
                    f"{source_id}: annotation fps {fps} does not match video fps "
                    f"{probed_fps}."
                )
            anchors, partners, pairs = sample_anchor_motion_frames(
                video_path, n=n, size=size, offset=offset
            )
            flows = dense_farneback_flows(anchors, partners, parameters)
            elapsed = np.asarray(
                [(second - first) / float(fps) for first, second in pairs],
                dtype=np.float32,
            )[:, None, None, None]
            magnitudes = np.linalg.norm(flows / elapsed, axis=-1).reshape(-1)
            finite = magnitudes[np.isfinite(magnitudes)]
            if finite.size == 0:
                raise ValueError(f"{source_id}: flow produced no finite velocities.")
            count = min(samples_per_source, int(finite.size))
            generator = np.random.default_rng(
                _source_sampling_seed(source_id, sampling_seed)
            )
            selected = generator.choice(finite.size, size=count, replace=False)
            samples[cursor : cursor + count] = finite[selected]
            cursor += count
            del anchors, partners, flows, magnitudes, finite
        samples.flush()
        if cursor == 0:
            raise ValueError("Training flow produced no velocity samples.")
        value = float(np.quantile(np.asarray(samples[:cursor]), quantile))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("Training flow did not produce a positive v_max.")
    return MotionScaleEstimate(
        v_max=value,
        sampled_values=cursor,
        source_count=len(sources),
        samples_per_source=samples_per_source,
        sampling_seed=sampling_seed,
    )


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
    """Stream only resized anchor/adjacent frames needed by the motion path."""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise VideoLoadError(f"Cannot open video file: {video_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    pairs = native_motion_pairs(total_frames, n=n, offset=offset)
    required = sorted({index for pair in pairs for index in pair})
    required_set = set(required)
    selected: dict[int, np.ndarray] = {}
    frame_index = 0
    try:
        while frame_index <= required[-1]:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in required_set:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                selected[frame_index] = cv2.resize(
                    rgb, size, interpolation=cv2.INTER_LINEAR
                )
            frame_index += 1
    finally:
        capture.release()
    missing = [index for index in required if index not in selected]
    if missing:
        raise VideoLoadError(
            f"Video '{video_path}' ended before required frame {missing[0]}."
        )
    anchors = [selected[first] for first, _ in pairs]
    partners = [selected[second] for _, second in pairs]
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
