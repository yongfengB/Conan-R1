"""Common data types for Conan-R1 / Surv-VAU."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEGRADATION_FACTORS = {
    "local_occlusion": ["vehicle_mask", "interaction_area_mask"],
    "evidence_quality": ["motion_blur", "lens_flare", "sensor_noise"],
    "environmental": ["low_light", "rain_snow", "fog", "tunnel_low_light"],
    # These operators are reserved for the synthetic-unseen test partition.
    "held_out_test": ["defocus_blur", "compression_artifact"],
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


class SpatialAnnotationError(ValueError):
    """Raised when a spatially targeted operator lacks track/region metadata."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrackBox:
    """A normalized bounding box attached to one source-video frame."""

    frame_index: int
    bbox: Tuple[float, float, float, float]

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise ValueError(
                "Track boxes must use normalized [x1, y1, x2, y2] coordinates"
            )


@dataclass
class ObjectTrack:
    """An object trajectory used by targeted occlusion and blur operators."""

    track_id: str
    category: str
    boxes: List[TrackBox]
    event_relevant: bool = True

    def __post_init__(self) -> None:
        if not self.track_id.strip() or not self.category.strip():
            raise ValueError("track_id and category must not be empty")
        if not self.boxes:
            raise ValueError("an object track must contain at least one box")
        ordered = sorted(self.boxes, key=lambda item: item.frame_index)
        if len({item.frame_index for item in ordered}) != len(ordered):
            raise ValueError("an object track cannot repeat a frame_index")
        self.boxes = ordered

    def box_at(self, frame_index: int) -> Optional[Tuple[float, float, float, float]]:
        """Linearly interpolate a box only while the track is visible."""
        if frame_index < self.boxes[0].frame_index or frame_index > self.boxes[-1].frame_index:
            return None
        for left, right in zip(self.boxes, self.boxes[1:]):
            if frame_index == left.frame_index:
                return left.bbox
            if left.frame_index < frame_index < right.frame_index:
                span = right.frame_index - left.frame_index
                alpha = (frame_index - left.frame_index) / span
                return tuple(
                    float(a + alpha * (b - a))
                    for a, b in zip(left.bbox, right.bbox)
                )
        return self.boxes[-1].bbox


@dataclass
class InteractionRegion:
    """A tracked region containing the interaction that defines the event."""

    region_id: str
    boxes: List[TrackBox]

    def __post_init__(self) -> None:
        if not self.region_id.strip() or not self.boxes:
            raise ValueError("an interaction region needs an id and boxes")
        ordered = sorted(self.boxes, key=lambda item: item.frame_index)
        if len({item.frame_index for item in ordered}) != len(ordered):
            raise ValueError("an interaction region cannot repeat a frame_index")
        self.boxes = ordered

    def box_at(self, frame_index: int) -> Optional[Tuple[float, float, float, float]]:
        proxy = ObjectTrack(self.region_id, "interaction", self.boxes)
        return proxy.box_at(frame_index)


@dataclass
class DegradationProfile:
    """Describes the controlled observation degradation applied to a clip."""
    factors: List[Tuple[str, float]] = field(default_factory=list)
    # Each element: (factor_name, severity)  severity in SEVERITY_LEVELS
    difficulty_level: float = 0.0  # one of {0.0, 0.2, 0.4, 0.8}
    domain: str = ""

    def __post_init__(self) -> None:
        if not self.domain:
            self.domain = "synthetic_seen" if self.factors else "clean"
        if self.domain not in {
            "clean", "synthetic_seen", "synthetic_unseen", "natural"
        }:
            raise ValueError(f"Unsupported generated degradation domain: {self.domain}")
        if self.difficulty_level not in SEVERITY_LEVELS:
            raise ValueError(
                f"difficulty_level must be one of {SEVERITY_LEVELS}, "
                f"got {self.difficulty_level}"
            )
        if not self.factors and self.difficulty_level != 0.0:
            raise ValueError("A non-zero degradation level requires factors")
        if self.factors and self.difficulty_level == 0.0:
            raise ValueError("The clean profile cannot contain degradation factors")
        if bool(self.factors) != self.domain.startswith("synthetic_"):
            raise ValueError("profile factors and generated domain are inconsistent")
        if len({factor_name for factor_name, _ in self.factors}) != len(self.factors):
            raise ValueError("A degradation profile cannot repeat an operator")
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
    task_mask: Dict[str, bool] = field(
        default_factory=lambda: {"event": True, "temporal": True}
    )
    influence_targets: dict = field(default_factory=dict)
    degradation_profiles: List[DegradationProfile] = field(default_factory=list)
    object_tracks: List[ObjectTrack] = field(default_factory=list)
    interaction_regions: List[InteractionRegion] = field(default_factory=list)
    fps: float = 30.0
    duration_sec: float = 0.0

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("A video clip must contain at least one frame")
        if self.start_frame >= self.end_frame:
            raise ValueError(
                f"start_frame ({self.start_frame}) must be < end_frame ({self.end_frame})"
            )
        if self.start_frame < 0 or self.end_frame >= len(self.frames):
            raise ValueError("Event frame bounds must lie inside the source clip")
        if not self.source_video_id:
            self.source_video_id = self.video_id
        if self.fps <= 0.0:
            raise ValueError("fps must be positive")
        if self.duration_sec <= 0.0:
            self.duration_sec = max(self.end_sec, len(self.frames) / self.fps)
        if not (0.0 <= self.start_sec < self.end_sec <= self.duration_sec):
            raise ValueError("Event seconds must lie inside the source duration")
        for track in self.object_tracks:
            if track.boxes[-1].frame_index >= len(self.frames):
                raise ValueError("Object-track frame indices must lie inside the clip")
        for region in self.interaction_regions:
            if region.boxes[-1].frame_index >= len(self.frames):
                raise ValueError("Interaction-region frame indices must lie inside the clip")


@dataclass
class DegradedClip:
    """A clip after controlled observation-degradation augmentation."""
    video_id: str
    frames: List[np.ndarray]
    start_sec: float
    end_sec: float
    profile: DegradationProfile
    source_clip: Optional[VideoClip] = None
    synthesis_metadata: dict = field(default_factory=dict)


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
    task_mask: Dict[str, bool] = field(
        default_factory=lambda: {"event": True, "temporal": True}
    )
    influence_targets: dict = field(default_factory=dict)
    synthesis_metadata: dict = field(default_factory=dict)
    degradation_domain: str = "synthetic_seen"

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
        if set(self.task_mask) != {"event", "temporal"}:
            raise ValueError("task_mask must contain exactly event and temporal")
        if not any(bool(value) for value in self.task_mask.values()):
            raise ValueError("At least one answer task must be active")
        if self.influence_targets and set(self.influence_targets) != {
            "affected_interval",
            "evidence_branch",
            "reliability_level",
            "cue_impact",
        }:
            raise ValueError("influence_targets has an invalid field set")
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
