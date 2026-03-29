"""Common data types for Conan-R1 / Surv-VAU."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIFFICULTY_FACTORS = {
    "local_occlusion": ["vehicle_mask", "interaction_area_mask"],
    "evidence_quality": ["motion_blur", "lens_flare", "sensor_noise"],
    "environmental": ["low_light", "rain_snow", "fog", "tunnel_low_light"],
}

SEVERITY_LEVELS = [0.0, 0.2, 0.4, 0.8]

VALID_FACTOR_NAMES: set = {
    f for factors in DIFFICULTY_FACTORS.values() for f in factors
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class VideoLoadError(Exception):
    """Raised when a video file cannot be loaded or has too few frames."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DegradationProfile:
    """Describes the observation-difficulty augmentation applied to a clip."""
    factors: List[Tuple[str, float]] = field(default_factory=list)
    # Each element: (factor_name, severity)  severity in SEVERITY_LEVELS
    difficulty_level: float = 0.0  # one of {0.0, 0.2, 0.4, 0.8}

    def __post_init__(self) -> None:
        if self.difficulty_level not in SEVERITY_LEVELS:
            raise ValueError(
                f"difficulty_level must be one of {SEVERITY_LEVELS}, "
                f"got {self.difficulty_level}"
            )

    def aggregated_score(self) -> float:
        """Mean severity across all active factors (s_bar)."""
        if not self.factors:
            return 0.0
        return sum(s for _, s in self.factors) / len(self.factors)


@dataclass
class VideoClip:
    """A source anomaly-event clip before degradation."""
    video_id: str
    frames: List[np.ndarray]          # raw frames (H x W x C)
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    source_path: str = ""

    def __post_init__(self) -> None:
        if self.start_frame >= self.end_frame:
            raise ValueError(
                f"start_frame ({self.start_frame}) must be < end_frame ({self.end_frame})"
            )


@dataclass
class DegradedClip:
    """A clip after observation-difficulty augmentation."""
    video_id: str
    frames: List[np.ndarray]
    start_sec: float
    end_sec: float
    profile: DegradationProfile
    source_clip: Optional[VideoClip] = None


@dataclass
class StructuredSample:
    """A fully annotated training/evaluation sample."""
    video_id: str
    frames: List[np.ndarray]
    prompt: str
    degradation_profile: List[Tuple[str, float]]  # [(factor, severity), ...]
    difficulty_level: float
    gt_interval: Tuple[float, float]              # (start_sec, end_sec)
    type_annotation: str
    influence_annotation: str
    reasoning_annotation: str
    conclusion_annotation: str
    answer_annotation: str
    split: str  # "sft_train" | "rl_train" | "val" | "test"

    def __post_init__(self) -> None:
        if self.difficulty_level not in SEVERITY_LEVELS:
            raise ValueError(
                f"difficulty_level must be one of {SEVERITY_LEVELS}"
            )
        if self.gt_interval[0] >= self.gt_interval[1]:
            raise ValueError("gt_interval start must be < end")
        for block_name, block_val in [
            ("type_annotation", self.type_annotation),
            ("influence_annotation", self.influence_annotation),
            ("reasoning_annotation", self.reasoning_annotation),
            ("conclusion_annotation", self.conclusion_annotation),
            ("answer_annotation", self.answer_annotation),
        ]:
            if not block_val or not block_val.strip():
                raise ValueError(f"{block_name} must not be empty")
