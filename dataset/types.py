"""Common data types for Conan-R1 / Surv-VAU."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEGRADATION_FACTORS = {
    "local_occlusion": ["vehicle_mask", "interaction_area_mask"],
    "evidence_quality": ["motion_blur", "lens_flare", "sensor_noise"],
    "environmental": ["low_light", "rain_snow", "fog", "tunnel_low_light"],
}

SEVERITY_LEVELS = [0.0, 0.2, 0.4, 0.8]

VALID_FACTOR_NAMES: set = {
    f for factors in DEGRADATION_FACTORS.values() for f in factors
}
# Backward-compatible name used by the initial data schema.
DIFFICULTY_FACTORS = DEGRADATION_FACTORS


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
    """Describes the controlled observation degradation applied to a clip."""
    factors: List[Tuple[str, float]] = field(default_factory=list)
    # Each element: (factor_name, severity)  severity in SEVERITY_LEVELS
    difficulty_level: float = 0.0  # one of {0.0, 0.2, 0.4, 0.8}

    def __post_init__(self) -> None:
        if self.difficulty_level not in SEVERITY_LEVELS:
            raise ValueError(
                f"difficulty_level must be one of {SEVERITY_LEVELS}, "
                f"got {self.difficulty_level}"
            )
        if not self.factors and self.difficulty_level != 0.0:
            raise ValueError("A non-zero degradation level requires factors")
        if self.factors and self.difficulty_level == 0.0:
            raise ValueError("The clean profile cannot contain degradation factors")
        for factor_name, severity in self.factors:
            if factor_name not in VALID_FACTOR_NAMES:
                raise ValueError(f"Unsupported degradation factor: {factor_name}")
            if float(severity) not in SEVERITY_LEVELS[1:]:
                raise ValueError(
                    f"factor severity must be one of {SEVERITY_LEVELS[1:]}, "
                    f"got {severity}"
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
    source_video_id: str = ""
    source_dataset: str = "unspecified"
    event_type: str = ""
    event_aliases: List[str] = field(default_factory=list)
    degradation_profiles: List[DegradationProfile] = field(default_factory=list)
    fps: float = 30.0
    duration_sec: float = 0.0

    def __post_init__(self) -> None:
        if self.start_frame >= self.end_frame:
            raise ValueError(
                f"start_frame ({self.start_frame}) must be < end_frame ({self.end_frame})"
            )
        if not self.source_video_id:
            self.source_video_id = self.video_id
        if self.fps <= 0.0:
            raise ValueError("fps must be positive")
        if self.duration_sec <= 0.0:
            self.duration_sec = max(self.end_sec, len(self.frames) / self.fps)


@dataclass
class DegradedClip:
    """A clip after controlled observation-degradation augmentation."""
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
    source_video_id: str
    source_dataset: str
    frames: List[np.ndarray]
    prompt: str
    degradation_profile: List[Tuple[str, float]]  # [(factor, severity), ...]
    difficulty_level: float
    gt_interval: Tuple[float, float]              # (start_sec, end_sec)
    event_type: str
    event_aliases: List[str]
    reasoning_target_length: int
    reasoning_target_source: str
    duration_sec: float
    fps: float
    num_source_frames: int
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
        if not self.source_video_id.strip():
            raise ValueError("source_video_id must not be empty")
        if not self.source_dataset.strip():
            raise ValueError("source_dataset must not be empty")
        if not self.event_type.strip():
            raise ValueError("event_type must not be empty")
        if self.reasoning_target_length <= 0:
            raise ValueError("reasoning_target_length must be positive")
        if self.reasoning_target_source not in {
            "human",
            "human_verified",
            "deterministic_policy",
        }:
            raise ValueError("reasoning_target_source is not auditable")
        if self.duration_sec <= 0.0:
            raise ValueError("duration_sec must be positive")
        if not (0.0 <= self.gt_interval[0] < self.gt_interval[1] <= self.duration_sec):
            raise ValueError("gt_interval must be within the video duration")
        if self.fps <= 0.0 or self.num_source_frames <= 1:
            raise ValueError("fps and num_source_frames must be positive")
        for block_name, block_val in [
            ("type_annotation", self.type_annotation),
            ("influence_annotation", self.influence_annotation),
            ("reasoning_annotation", self.reasoning_annotation),
            ("conclusion_annotation", self.conclusion_annotation),
            ("answer_annotation", self.answer_annotation),
        ]:
            if not block_val or not block_val.strip():
                raise ValueError(f"{block_name} must not be empty")
