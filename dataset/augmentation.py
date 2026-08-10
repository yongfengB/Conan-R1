"""Object-aware and temporally coherent observation degradation for Surv-VAU.

All stochastic state is sampled once per video/profile pair.  Spatially
targeted operators consume normalized object tracks or interaction regions;
they never fall back to a fixed image rectangle.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .types import (
    DegradationProfile,
    DegradedClip,
    InteractionRegion,
    ObjectTrack,
    SpatialAnnotationError,
    VideoClip,
)


BBox = Tuple[float, float, float, float]
OPERATOR_ORDER = [
    "vehicle_mask",
    "interaction_area_mask",
    "motion_blur",
    "defocus_blur",
    "low_light",
    "tunnel_low_light",
    "fog",
    "rain_snow",
    "lens_flare",
    "sensor_noise",
    "compression_artifact",
]
_OPERATOR_RANK = {name: index for index, name in enumerate(OPERATOR_ORDER)}


def _odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def _stable_seed(clip: VideoClip, profile: DegradationProfile, seed: int) -> int:
    factors = ";".join(f"{name}:{severity:.3f}" for name, severity in profile.factors)
    payload = f"{seed}|{clip.source_video_id}|{clip.video_id}|{factors}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _normalized_box_to_pixels(box: BBox, height: int, width: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(width - 1, int(round(x1 * width)))),
        max(0, min(height - 1, int(round(y1 * height)))),
        max(1, min(width, int(round(x2 * width)))),
        max(1, min(height, int(round(y2 * height)))),
    )


def _box_center(box: BBox) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _union_box(first: BBox, second: BBox, margin: float = 0.0) -> BBox:
    return (
        max(0.0, min(first[0], second[0]) - margin),
        max(0.0, min(first[1], second[1]) - margin),
        min(1.0, max(first[2], second[2]) + margin),
        min(1.0, max(first[3], second[3]) + margin),
    )


def _event_frame_range(clip: VideoClip) -> Tuple[int, int]:
    start = max(0, int(clip.start_frame))
    end = min(len(clip.frames) - 1, int(clip.end_frame))
    return start, max(start, end)


def _select_vehicle_track(clip: VideoClip) -> ObjectTrack:
    vehicles = [
        track
        for track in clip.object_tracks
        if track.category.lower() in {"vehicle", "car", "truck", "bus", "van", "motorcycle"}
    ]
    if not vehicles:
        raise SpatialAnnotationError(
            f"{clip.video_id}: vehicle_mask requires at least one vehicle trajectory"
        )
    start, end = _event_frame_range(clip)

    def score(track: ObjectTrack) -> Tuple[int, int, float]:
        visible = [
            box for frame_index in range(start, end + 1)
            if (box := track.box_at(frame_index)) is not None
        ]
        mean_area = (
            sum((box[2] - box[0]) * (box[3] - box[1]) for box in visible)
            / max(1, len(visible))
        )
        return int(track.event_relevant), len(visible), mean_area

    return max(vehicles, key=score)


def _select_interaction_source(
    clip: VideoClip,
) -> Tuple[Optional[InteractionRegion], Optional[Tuple[ObjectTrack, ObjectTrack]]]:
    if clip.interaction_regions:
        start, end = _event_frame_range(clip)
        region = max(
            clip.interaction_regions,
            key=lambda item: sum(
                item.box_at(frame_index) is not None
                for frame_index in range(start, end + 1)
            ),
        )
        return region, None

    tracks = [track for track in clip.object_tracks if track.event_relevant]
    if len(tracks) < 2:
        raise SpatialAnnotationError(
            f"{clip.video_id}: interaction_area_mask requires a tracked interaction "
            "region or at least two event-relevant object trajectories"
        )
    start, end = _event_frame_range(clip)
    best_pair: Optional[Tuple[ObjectTrack, ObjectTrack]] = None
    best_distance = float("inf")
    for first_index, first in enumerate(tracks):
        for second in tracks[first_index + 1 :]:
            distances = []
            for frame_index in range(start, end + 1):
                first_box = first.box_at(frame_index)
                second_box = second.box_at(frame_index)
                if first_box is None or second_box is None:
                    continue
                first_center = _box_center(first_box)
                second_center = _box_center(second_box)
                distances.append(math.dist(first_center, second_center))
            if distances and float(np.mean(distances)) < best_distance:
                best_distance = float(np.mean(distances))
                best_pair = (first, second)
    if best_pair is None:
        raise SpatialAnnotationError(
            f"{clip.video_id}: no overlapping trajectories define an interaction area"
        )
    return None, best_pair


def _dominant_track_angle(clip: VideoClip) -> float:
    displacements = []
    for track in clip.object_tracks:
        for first, second in zip(track.boxes, track.boxes[1:]):
            first_center = _box_center(first.bbox)
            second_center = _box_center(second.bbox)
            dx = second_center[0] - first_center[0]
            dy = second_center[1] - first_center[1]
            if abs(dx) + abs(dy) > 1e-6:
                displacements.append((dx, dy))
    if not displacements:
        return 0.0
    dx = float(np.median([item[0] for item in displacements]))
    dy = float(np.median([item[1] for item in displacements]))
    return math.degrees(math.atan2(dy, dx))


def apply_motion_blur(
    frame: np.ndarray,
    severity: float,
    angle_degrees: float = 0.0,
) -> np.ndarray:
    """Apply a line-spread blur, capped at a 31-pixel kernel."""
    kernel_size = _odd(max(1, int(round(severity * 30.0))))
    if kernel_size == 1:
        return frame.copy()
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    rotation = cv2.getRotationMatrix2D(
        (kernel_size / 2.0 - 0.5, kernel_size / 2.0 - 0.5), angle_degrees, 1.0
    )
    kernel = cv2.warpAffine(kernel, rotation, (kernel_size, kernel_size))
    kernel /= max(float(kernel.sum()), 1e-8)
    return cv2.filter2D(frame, -1, kernel)


def apply_lens_flare(
    frame: np.ndarray,
    severity: float,
    center: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Overlay a flare with radius capped at 40% of the shorter image side."""
    out = frame.copy().astype(np.float32)
    height, width = frame.shape[:2]
    center = center or (0.7, 0.3)
    cx = int(np.clip(center[0], 0.0, 1.0) * (width - 1))
    cy = int(np.clip(center[1], 0.0, 1.0) * (height - 1))
    radius = int(round(min(height, width) * severity * 0.4))
    if radius < 1:
        return frame.copy()
    overlay = np.zeros_like(out)
    cv2.circle(overlay, (cx, cy), radius, (255.0, 245.0, 215.0), -1)
    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=max(1.0, radius / 3.0))
    alpha = min(0.65, 0.65 * severity)
    return np.clip(out + alpha * overlay, 0, 255).astype(np.uint8)


def apply_sensor_noise(
    frame: np.ndarray,
    severity: float,
    noise_field: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Add zero-mean sensor noise with standard deviation capped at 50 levels."""
    if noise_field is None:
        generator = rng or np.random.default_rng(0)
        noise_field = generator.normal(0.0, 1.0, frame.shape).astype(np.float32)
    noisy = frame.astype(np.float32) + noise_field * (severity * 50.0)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_low_light(frame: np.ndarray, severity: float) -> np.ndarray:
    """Reduce intensity by at most 80%, preserving the original color ratios."""
    factor = 1.0 - severity * 0.8
    return np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def apply_rain_snow(
    frame: np.ndarray,
    severity: float,
    particles: Optional[np.ndarray] = None,
    frame_index: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Overlay up to 300 persistent, advected rain/snow particles."""
    out = frame.copy()
    height, width = frame.shape[:2]
    count = int(round(severity * 300.0))
    if count <= 0:
        return out
    if particles is None:
        generator = rng or np.random.default_rng(0)
        particles = np.column_stack(
            [
                generator.random(count),
                generator.random(count),
                generator.uniform(-0.002, 0.002, count),
                generator.uniform(0.012, 0.025, count),
                generator.uniform(0.025, 0.075, count),
                generator.uniform(180.0, 255.0, count),
            ]
        )
    for x0, y0, vx, vy, length, brightness in particles[:count]:
        x = (x0 + vx * frame_index) % 1.0
        y = (y0 + vy * frame_index) % 1.0
        x1, y1 = int(x * width), int(y * height)
        x2 = int((x + vx * 1.8) * width)
        y2 = int((y + length) * height)
        value = int(brightness)
        cv2.line(out, (x1, y1), (x2, y2), (value, value, value), 1)
    return out


def apply_fog(frame: np.ndarray, severity: float) -> np.ndarray:
    """Blend a neutral veil with opacity capped at 70%."""
    fog_layer = np.full_like(frame, 220, dtype=np.uint8)
    alpha = severity * 0.7
    return cv2.addWeighted(frame, 1.0 - alpha, fog_layer, alpha, 0)


def apply_defocus_blur(frame: np.ndarray, severity: float) -> np.ndarray:
    """Held-out test operator: Gaussian defocus with sigma capped at 4 pixels."""
    sigma = max(0.0, severity * 4.0)
    if sigma < 1e-8:
        return frame.copy()
    return cv2.GaussianBlur(frame, (0, 0), sigmaX=sigma, sigmaY=sigma)


def apply_compression_artifact(frame: np.ndarray, severity: float) -> np.ndarray:
    """Held-out test operator: JPEG quality decreases from 100 to a floor of 20."""
    quality = max(20, int(round(100.0 - severity * 80.0)))
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed while synthesizing compression artifacts")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG decoding failed while synthesizing compression artifacts")
    return decoded


def apply_occlusion(
    frame: np.ndarray,
    severity: float,
    bbox: BBox,
    opacity: float = 0.95,
) -> np.ndarray:
    """Occlude a severity-controlled fraction of an annotated/tracked region."""
    out = frame.copy().astype(np.float32)
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    center_x, center_y = _box_center(bbox)
    # Cover `severity` of the target area; s=1 covers the complete target box.
    scale = math.sqrt(max(0.0, min(1.0, severity)))
    half_width = (x2 - x1) * scale / 2.0
    half_height = (y2 - y1) * scale / 2.0
    target = (
        max(0.0, center_x - half_width),
        max(0.0, center_y - half_height),
        min(1.0, center_x + half_width),
        min(1.0, center_y + half_height),
    )
    px1, py1, px2, py2 = _normalized_box_to_pixels(target, height, width)
    if px2 <= px1 or py2 <= py1:
        return frame.copy()
    fill = np.full_like(out[py1:py2, px1:px2], 16.0)
    alpha = max(0.0, min(1.0, opacity))
    out[py1:py2, px1:px2] = (
        (1.0 - alpha) * out[py1:py2, px1:px2] + alpha * fill
    )
    return np.clip(out, 0, 255).astype(np.uint8)


@dataclass
class _TemporalContext:
    clip: VideoClip
    profile: DegradationProfile
    seed: int

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        height, width = self.clip.frames[0].shape[:2]
        self.shape = (height, width, self.clip.frames[0].shape[2])
        self.blur_angle = _dominant_track_angle(self.clip)
        self.flare_origin = (
            float(self.rng.uniform(0.58, 0.78)),
            float(self.rng.uniform(0.18, 0.38)),
        )
        self.flare_velocity = (
            float(self.rng.uniform(-0.0025, 0.0025)),
            float(self.rng.uniform(-0.0015, 0.0015)),
        )
        self.noise_state = self.rng.normal(0.0, 1.0, self.shape).astype(np.float32)
        max_particles = 300
        self.weather_particles = np.column_stack(
            [
                self.rng.random(max_particles),
                self.rng.random(max_particles),
                self.rng.uniform(-0.002, 0.002, max_particles),
                self.rng.uniform(0.012, 0.025, max_particles),
                self.rng.uniform(0.025, 0.075, max_particles),
                self.rng.uniform(180.0, 255.0, max_particles),
            ]
        )
        active = {name for name, _ in self.profile.factors}
        self.vehicle_track = _select_vehicle_track(self.clip) if "vehicle_mask" in active else None
        if "interaction_area_mask" in active:
            self.interaction_region, self.interaction_pair = _select_interaction_source(self.clip)
        else:
            self.interaction_region, self.interaction_pair = None, None

    def flare_center(self, frame_index: int) -> Tuple[float, float]:
        phase = 2.0 * math.pi * frame_index / max(1, len(self.clip.frames) - 1)
        return (
            self.flare_origin[0] + self.flare_velocity[0] * frame_index + 0.01 * math.sin(phase),
            self.flare_origin[1] + self.flare_velocity[1] * frame_index + 0.006 * math.cos(phase),
        )

    def next_noise(self) -> np.ndarray:
        innovation = self.rng.normal(0.0, 1.0, self.shape).astype(np.float32)
        rho = 0.85
        self.noise_state = rho * self.noise_state + math.sqrt(1.0 - rho * rho) * innovation
        return self.noise_state

    def occlusion_box(self, factor: str, frame_index: int) -> Optional[BBox]:
        if factor == "vehicle_mask":
            return self.vehicle_track.box_at(frame_index) if self.vehicle_track else None
        if self.interaction_region is not None:
            return self.interaction_region.box_at(frame_index)
        if self.interaction_pair is not None:
            first_box = self.interaction_pair[0].box_at(frame_index)
            second_box = self.interaction_pair[1].box_at(frame_index)
            if first_box is not None and second_box is not None:
                return _union_box(first_box, second_box, margin=0.015)
        return None


def _apply_factor(
    frame: np.ndarray,
    factor: str,
    severity: float,
    frame_index: int,
    context: _TemporalContext,
) -> np.ndarray:
    if factor == "motion_blur":
        phase = 2.0 * math.pi * frame_index / max(1, len(context.clip.frames) - 1)
        return apply_motion_blur(frame, severity, context.blur_angle + 5.0 * math.sin(phase))
    if factor == "lens_flare":
        return apply_lens_flare(frame, severity, context.flare_center(frame_index))
    if factor == "sensor_noise":
        return apply_sensor_noise(frame, severity, context.next_noise())
    if factor in {"low_light", "tunnel_low_light"}:
        return apply_low_light(frame, severity)
    if factor == "rain_snow":
        return apply_rain_snow(
            frame, severity, context.weather_particles, frame_index=frame_index
        )
    if factor == "fog":
        return apply_fog(frame, severity)
    if factor in {"vehicle_mask", "interaction_area_mask"}:
        bbox = context.occlusion_box(factor, frame_index)
        return frame.copy() if bbox is None else apply_occlusion(frame, severity, bbox)
    if factor == "defocus_blur":
        return apply_defocus_blur(frame, severity)
    if factor == "compression_artifact":
        return apply_compression_artifact(frame, severity)
    raise ValueError(f"Unsupported degradation factor: {factor}")


def synthesize_degradation(
    clip: VideoClip,
    profile: DegradationProfile,
    seed: int = 42,
) -> DegradedClip:
    """Apply a deterministic, temporally coherent profile to an entire clip.

    Local occlusion is defined only by object trajectories or interaction-region
    annotations.  Weather particles, flare position, noise state and blur angle
    evolve from video-level state rather than being independently resampled per
    frame.
    """
    if not clip.frames:
        raise ValueError("Cannot synthesize degradation for an empty clip")
    resolved_seed = _stable_seed(clip, profile, seed)
    context = _TemporalContext(clip=clip, profile=profile, seed=resolved_seed)
    degraded_frames: List[np.ndarray] = []
    ordered_factors = sorted(
        profile.factors, key=lambda item: _OPERATOR_RANK[item[0]]
    )
    for frame_index, frame in enumerate(clip.frames):
        out = frame.copy()
        for factor_name, severity in ordered_factors:
            out = _apply_factor(out, factor_name, severity, frame_index, context)
        degraded_frames.append(out)

    return DegradedClip(
        video_id=clip.video_id,
        frames=degraded_frames,
        start_sec=clip.start_sec,
        end_sec=clip.end_sec,
        profile=profile,
        source_clip=clip,
        synthesis_metadata={
            "protocol": "surv-vau-degradation-v1",
            "seed": resolved_seed,
            "temporal_state": "video_level",
            "spatial_targeting": "tracks_or_interaction_regions",
        },
    )


def synthesize_difficulty(
    clip: VideoClip,
    profile: DegradationProfile,
    seed: int = 42,
) -> DegradedClip:
    """Backward-compatible alias for :func:`synthesize_degradation`."""
    return synthesize_degradation(clip, profile, seed=seed)
