"""Observation-difficulty augmentation functions for Surv-VAU."""
from __future__ import annotations
from typing import List, Tuple

import cv2
import numpy as np

from .types import DegradationProfile, DegradedClip, VideoClip, SEVERITY_LEVELS


# ---------------------------------------------------------------------------
# Individual degradation functions
# ---------------------------------------------------------------------------

def apply_motion_blur(frame: np.ndarray, severity: float) -> np.ndarray:
    """Apply motion blur proportional to severity (0-1)."""
    kernel_size = max(1, int(severity * 30))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    return cv2.filter2D(frame, -1, kernel)


def apply_lens_flare(frame: np.ndarray, severity: float) -> np.ndarray:
    """Overlay a synthetic lens flare proportional to severity."""
    out = frame.copy().astype(np.float32)
    h, w = frame.shape[:2]
    cx, cy = int(w * 0.7), int(h * 0.3)
    radius = int(min(h, w) * severity * 0.4)
    if radius < 1:
        return frame
    for r in range(radius, 0, -max(1, radius // 5)):
        alpha = severity * (1 - r / radius) * 200
        cv2.circle(out, (cx, cy), r, (alpha, alpha, alpha * 0.8), -1)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_sensor_noise(frame: np.ndarray, severity: float) -> np.ndarray:
    """Add Gaussian noise proportional to severity."""
    sigma = severity * 50.0
    noise = np.random.normal(0, sigma, frame.shape).astype(np.float32)
    noisy = frame.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_low_light(frame: np.ndarray, severity: float) -> np.ndarray:
    """Darken the frame proportional to severity."""
    factor = 1.0 - severity * 0.8
    darkened = (frame.astype(np.float32) * factor)
    return np.clip(darkened, 0, 255).astype(np.uint8)


def apply_rain_snow(frame: np.ndarray, severity: float) -> np.ndarray:
    """Overlay rain/snow streaks proportional to severity."""
    out = frame.copy()
    h, w = frame.shape[:2]
    n_streaks = int(severity * 300)
    for _ in range(n_streaks):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        length = np.random.randint(5, 20)
        angle = np.random.uniform(-20, 20)
        dx = int(length * np.sin(np.radians(angle)))
        dy = int(length * np.cos(np.radians(angle)))
        brightness = np.random.randint(180, 255)
        cv2.line(out, (x, y), (x + dx, y + dy), (brightness,) * 3, 1)
    return out


def apply_fog(frame: np.ndarray, severity: float) -> np.ndarray:
    """Blend frame with a white fog layer proportional to severity."""
    fog_layer = np.full_like(frame, 220, dtype=np.uint8)
    alpha = severity * 0.7
    blended = cv2.addWeighted(frame, 1 - alpha, fog_layer, alpha, 0)
    return blended


def apply_occlusion(
    frame: np.ndarray,
    severity: float,
    factor_name: str = "vehicle_mask",
) -> np.ndarray:
    """Mask a rectangular region to simulate occlusion."""
    out = frame.copy()
    h, w = frame.shape[:2]
    mask_h = int(h * severity * 0.5)
    mask_w = int(w * severity * 0.5)
    if mask_h < 1 or mask_w < 1:
        return out
    # Place mask in center for vehicle_mask, top-left for interaction_area_mask
    if factor_name == "interaction_area_mask":
        y1, x1 = 0, 0
    else:
        y1 = (h - mask_h) // 2
        x1 = (w - mask_w) // 2
    y2, x2 = y1 + mask_h, x1 + mask_w
    out[y1:y2, x1:x2] = 0
    return out


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_FACTOR_FN = {
    "motion_blur": apply_motion_blur,
    "lens_flare": apply_lens_flare,
    "sensor_noise": apply_sensor_noise,
    "low_light": apply_low_light,
    "rain_snow": apply_rain_snow,
    "fog": apply_fog,
    "tunnel_low_light": apply_low_light,
    "vehicle_mask": lambda f, s: apply_occlusion(f, s, "vehicle_mask"),
    "interaction_area_mask": lambda f, s: apply_occlusion(f, s, "interaction_area_mask"),
}


def _apply_factor(frame: np.ndarray, factor: str, severity: float) -> np.ndarray:
    fn = _FACTOR_FN.get(factor)
    if fn is None:
        return frame
    return fn(frame, severity)


# ---------------------------------------------------------------------------
# Main synthesis function
# ---------------------------------------------------------------------------

def synthesize_difficulty(
    clip: VideoClip,
    profile: DegradationProfile,
) -> DegradedClip:
    """Apply all degradation factors in `profile` to every frame of `clip`.

    Args:
        clip: Source VideoClip.
        profile: DegradationProfile specifying factors and severities.

    Returns:
        DegradedClip with augmented frames.

    Raises:
        ValueError: If profile.difficulty_level is not in SEVERITY_LEVELS.
    """
    # Validation is done in DegradationProfile.__post_init__
    degraded_frames: List[np.ndarray] = []
    for frame in clip.frames:
        out = frame.copy()
        for factor_name, severity in profile.factors:
            out = _apply_factor(out, factor_name, severity)
        degraded_frames.append(out)

    return DegradedClip(
        video_id=clip.video_id,
        frames=degraded_frames,
        start_sec=clip.start_sec,
        end_sec=clip.end_sec,
        profile=profile,
        source_clip=clip,
    )
