"""SurvVAU dataset builder — five-stage pipeline."""
from __future__ import annotations
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .annotation_pipeline import (
    adjust_compactness,
    generate_influence,
    generate_reasoning,
)
from .augmentation import synthesize_degradation
from .degradation_protocol import (
    load_degradation_protocol,
    validate_profile_compatibility,
)
from .types import (
    SEVERITY_LEVELS,
    DegradationProfile,
    DegradedClip,
    InteractionRegion,
    ObjectTrack,
    StructuredSample,
    TrackBox,
    VideoClip,
    VideoLoadError,
)
from .video_utils import frames_from_array, load_video
from .splitting import stratified_partition

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "Describe the traffic anomaly event shown in this surveillance video clip "
    "and identify its temporal boundaries [start_sec, end_sec]."
)


class SurvVAUBuilder:
    """Orchestrates the five-stage Surv-VAU construction pipeline."""

    def __init__(self, annotator_model: Any, seed: int = 42) -> None:
        self.model_q = annotator_model
        self.seed = seed
        random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass

    @staticmethod
    def serialize_answer(
        event_type: str,
        interval: Tuple[float, float],
        task_mask: Optional[Dict[str, bool]] = None,
    ) -> str:
        """Serialize the two independently annotated benchmark fields once.

        Model-authored prose is deliberately excluded from ``<ANSWER>``.  It
        remains available in ``<REASONING>`` and ``<CONCLUSION>``.  Keeping the
        benchmark block canonical prevents a generated explanation from
        introducing a second time interval that would make parsing ambiguous.
        """
        normalized_event = " ".join(str(event_type).split())
        if not normalized_event or any(token in normalized_event for token in (";", "[", "]")):
            raise ValueError("event_type contains a reserved ANSWER delimiter")
        start, end = map(float, interval)
        if not (0.0 <= start < end):
            raise ValueError("ANSWER interval must satisfy 0 <= start < end")
        active = task_mask or {"event": True, "temporal": True}
        fields = []
        if active.get("event", True):
            fields.append(f"event_type: {normalized_event}")
        if active.get("temporal", True):
            fields.append(f"interval: [{start:.3f}, {end:.3f}]")
        if not fields:
            raise ValueError("At least one ANSWER field must be active")
        return "; ".join(fields)

    # ------------------------------------------------------------------
    # Stage 1: collect & segment
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_degradation_profiles(
        raw_profiles: Any,
        video_id: str,
    ) -> List[DegradationProfile]:
        """Parse the frozen per-source profile manifest.

        Profiles are specified by the release annotation rather than created
        by a hidden hard-coded factor pair. This makes the code reproduce the
        exact clean, single-factor, and compound conditions described in the
        paper.
        """
        if not isinstance(raw_profiles, list) or not raw_profiles:
            raise ValueError(
                f"{video_id}: degradation_profiles must be a non-empty list"
            )
        profiles: List[DegradationProfile] = []
        for profile_index, raw_profile in enumerate(raw_profiles):
            if not isinstance(raw_profile, dict):
                raise ValueError(
                    f"{video_id}: profile {profile_index} must be an object"
                )
            raw_factors = raw_profile.get("factors", [])
            if not isinstance(raw_factors, list):
                raise ValueError(
                    f"{video_id}: profile {profile_index} factors must be a list"
                )
            factors: List[Tuple[str, float]] = []
            for factor in raw_factors:
                if isinstance(factor, dict):
                    name = str(factor.get("name", "")).strip()
                    severity = float(factor.get("severity"))
                elif isinstance(factor, (list, tuple)) and len(factor) == 2:
                    name = str(factor[0]).strip()
                    severity = float(factor[1])
                else:
                    raise ValueError(
                        f"{video_id}: invalid factor in profile {profile_index}"
                    )
                factors.append((name, severity))
            level = float(raw_profile.get("degradation_level", 0.0))
            profiles.append(
                DegradationProfile(
                    factors=factors,
                    difficulty_level=level,
                    domain=str(
                        raw_profile.get(
                            "degradation_domain",
                            "synthetic_seen" if factors else "clean",
                        )
                    ),
                )
            )
        natural_profiles = [profile for profile in profiles if profile.domain == "natural"]
        if natural_profiles and (
            len(profiles) != 1 or natural_profiles[0].factors
        ):
            raise ValueError(
                f"{video_id}: a natural source observation must be a single "
                "non-synthetic profile, not a clean/synthetic duplicate"
            )
        if not natural_profiles and not any(not profile.factors for profile in profiles):
            raise ValueError(f"{video_id}: a clean 0% profile is required")
        return profiles

    @staticmethod
    def _parse_track_boxes(raw_boxes: Any, owner: str) -> List[TrackBox]:
        if not isinstance(raw_boxes, list) or not raw_boxes:
            raise ValueError(f"{owner}: boxes must be a non-empty list")
        boxes = []
        for raw_box in raw_boxes:
            if not isinstance(raw_box, dict):
                raise ValueError(f"{owner}: each tracked box must be an object")
            coordinates = raw_box.get("bbox_norm", raw_box.get("bbox"))
            if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 4:
                raise ValueError(f"{owner}: bbox_norm must contain four values")
            boxes.append(
                TrackBox(
                    frame_index=int(raw_box["frame_index"]),
                    bbox=tuple(float(value) for value in coordinates),
                )
            )
        return boxes

    @classmethod
    def _parse_object_tracks(cls, raw_tracks: Any, video_id: str) -> List[ObjectTrack]:
        if raw_tracks is None:
            return []
        if not isinstance(raw_tracks, list):
            raise ValueError(f"{video_id}: object_tracks must be a list")
        tracks = []
        for raw_track in raw_tracks:
            if not isinstance(raw_track, dict):
                raise ValueError(f"{video_id}: each object track must be an object")
            track_id = str(raw_track.get("track_id", "")).strip()
            tracks.append(
                ObjectTrack(
                    track_id=track_id,
                    category=str(raw_track.get("category", "")).strip(),
                    event_relevant=bool(raw_track.get("event_relevant", True)),
                    boxes=cls._parse_track_boxes(
                        raw_track.get("boxes"), f"{video_id}:{track_id}"
                    ),
                )
            )
        return tracks

    @classmethod
    def _parse_interaction_regions(
        cls, raw_regions: Any, video_id: str
    ) -> List[InteractionRegion]:
        if raw_regions is None:
            return []
        if not isinstance(raw_regions, list):
            raise ValueError(f"{video_id}: interaction_regions must be a list")
        regions = []
        for raw_region in raw_regions:
            if not isinstance(raw_region, dict):
                raise ValueError(f"{video_id}: each interaction region must be an object")
            region_id = str(raw_region.get("region_id", "")).strip()
            regions.append(
                InteractionRegion(
                    region_id=region_id,
                    boxes=cls._parse_track_boxes(
                        raw_region.get("boxes"), f"{video_id}:{region_id}"
                    ),
                )
            )
        return regions

    def collect_and_segment(
        self,
        source_dirs: List[str],
        annotation_file: str,
    ) -> List[VideoClip]:
        """Load source videos and wrap them as VideoClip objects.

        Args:
            source_dirs: Directories containing source video files.
            annotation_file: JSON file with independent event/time annotations
                and a frozen ``degradation_profiles`` list for every source.

        Returns:
            List of VideoClip objects.
        """
        with open(annotation_file) as f:
            annotations: Dict = json.load(f)

        clips: List[VideoClip] = []
        for src_dir in source_dirs:
            for video_path in sorted(Path(src_dir).glob("*.mp4")):
                video_id = video_path.stem
                try:
                    frames = load_video(str(video_path))
                except VideoLoadError as e:
                    logger.error("Skipping %s: %s", video_path, e)
                    continue

                ann = annotations.get(video_id, {})
                event_type = str(ann.get("event_type", "")).strip()
                if not event_type:
                    logger.error(
                        "Skipping %s: independent event_type annotation is required.",
                        video_id,
                    )
                    continue
                start_frame = ann.get("start_frame", 0)
                end_frame = ann.get("end_frame", len(frames) - 1)
                start_sec = ann.get("start_sec", 0.0)
                fps = float(ann.get("fps", 30.0))
                duration_sec = float(ann.get("duration_sec", len(frames) / fps))
                end_sec = ann.get("end_sec", duration_sec)

                try:
                    degradation_profiles = self._parse_degradation_profiles(
                        ann.get("degradation_profiles"), video_id
                    )
                    task_mask = dict(
                        ann.get("task_mask", {"event": True, "temporal": True})
                    )
                    if set(task_mask) != {"event", "temporal"} or not any(
                        bool(value) for value in task_mask.values()
                    ):
                        raise ValueError(
                            "task_mask must define event/temporal with one active task"
                        )
                    influence_targets = dict(ann.get("influence_targets", {}))
                    if not influence_targets:
                        raise ValueError(
                            "influence_targets is required; the builder will not "
                            "invent reliability supervision"
                        )
                    clip = VideoClip(
                        video_id=video_id,
                        frames=frames,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        source_path=str(video_path),
                        source_video_id=str(
                            ann.get("source_video_id", video_id)
                        ),
                        source_dataset=str(
                            ann.get("source_dataset", "unspecified")
                        ),
                        scene_environment=str(
                            ann.get("scene_environment", "")
                        ),
                        event_type=event_type,
                        event_aliases=list(ann.get("event_aliases", [])),
                        task_mask=task_mask,
                        influence_targets=influence_targets,
                        degradation_profiles=degradation_profiles,
                        object_tracks=self._parse_object_tracks(
                            ann.get("object_tracks"), video_id
                        ),
                        interaction_regions=self._parse_interaction_regions(
                            ann.get("interaction_regions"), video_id
                        ),
                        fps=fps,
                        duration_sec=duration_sec,
                    )
                    clips.append(clip)
                except ValueError as e:
                    logger.error("Invalid clip %s: %s", video_id, e)

        logger.info("Collected %d clips from %d directories.", len(clips), len(source_dirs))
        return clips

    # ------------------------------------------------------------------
    # Stage 2: controlled degradation synthesis
    # ------------------------------------------------------------------

    @staticmethod
    def _build_profiles(clip: VideoClip) -> List[DegradationProfile]:
        """Return the source-specific frozen degradation profiles."""
        if not clip.degradation_profiles:
            raise ValueError(
                f"{clip.video_id}: no frozen degradation profiles are available"
            )
        protocol = load_degradation_protocol()
        for profile in clip.degradation_profiles:
            validate_profile_compatibility(
                profile, clip.scene_environment, protocol
            )
        return clip.degradation_profiles

    @staticmethod
    def _profile_suffix(profile: DegradationProfile) -> str:
        if not profile.factors:
            return "natural" if profile.domain == "natural" else "clean"
        return "__".join(
            f"{factor}-{int(round(severity * 100)):02d}"
            for factor, severity in profile.factors
        )

    # ------------------------------------------------------------------
    # Stage 3-4: annotation generation
    # ------------------------------------------------------------------

    def _annotate(
        self,
        degraded: DegradedClip,
        profile: DegradationProfile,
    ) -> Optional[StructuredSample]:
        """Run stages 3-5 to produce a StructuredSample."""
        try:
            influence = generate_influence(degraded, profile, self.model_q)
            reasoning, conclusion = generate_reasoning(
                degraded, profile, influence, self.model_q
            )
            # Stage 5: compactness adjustment
            reasoning_adj = adjust_compactness(reasoning, self.model_q)

            type_annotation = "; ".join(
                f"{name}:{sev:.1f}" for name, sev in profile.factors
            ) or "none"

            sample = StructuredSample(
                video_id=degraded.video_id,
                source_video_id=degraded.source_clip.source_video_id,
                source_dataset=degraded.source_clip.source_dataset,
                scene_environment=degraded.source_clip.scene_environment,
                frames=degraded.frames,
                prompt=DEFAULT_PROMPT,
                degradation_profile=profile.factors,
                difficulty_level=profile.difficulty_level,
                gt_interval=(degraded.start_sec, degraded.end_sec),
                event_type=degraded.source_clip.event_type,
                event_aliases=degraded.source_clip.event_aliases,
                task_mask=degraded.source_clip.task_mask,
                influence_targets=degraded.source_clip.influence_targets,
                duration_sec=degraded.source_clip.duration_sec,
                fps=degraded.source_clip.fps,
                num_source_frames=len(degraded.source_clip.frames),
                type_annotation=type_annotation,
                influence_annotation=influence,
                reasoning_annotation=reasoning_adj,
                conclusion_annotation=conclusion,
                answer_annotation=self.serialize_answer(
                    degraded.source_clip.event_type,
                    (degraded.start_sec, degraded.end_sec),
                    degraded.source_clip.task_mask,
                ),
                split="",  # assigned later
                synthesis_metadata=degraded.synthesis_metadata,
                degradation_domain=profile.domain,
            )
            return sample
        except Exception as e:
            logger.error("Annotation failed for %s: %s", degraded.video_id, e)
            return None

    # ------------------------------------------------------------------
    # Stage 5: split & quality filter
    # ------------------------------------------------------------------

    def split_dataset(
        self,
        samples: List[StructuredSample],
        seed: Optional[int] = None,
    ) -> Dict[str, List[StructuredSample]]:
        """Split at source-video level: 70% train, 15% val, 15% test.

        All augmented variants of the same source video go to the same split.
        Training is further divided: 30% SFT, 70% RL.
        """
        split_seed = seed if seed is not None else self.seed
        by_source = defaultdict(list)
        for sample in samples:
            by_source[sample.source_video_id].append(
                {
                    "source_dataset": sample.source_dataset,
                    "event_type": sample.event_type,
                }
            )
        outer = stratified_partition(
            by_source,
            fractions=(0.70, 0.15, 0.15),
            names=("train", "val", "test"),
            seed=split_seed,
        )
        training = {
            source_id: records
            for source_id, records in by_source.items()
            if outer[source_id] == "train"
        }
        inner = stratified_partition(
            training,
            fractions=(0.30, 0.70),
            names=("sft_train", "rl_train"),
            seed=split_seed + 1,
        )
        source_split = {
            source_id: inner[source_id] if split == "train" else split
            for source_id, split in outer.items()
        }

        splits: Dict[str, List[StructuredSample]] = {
            "sft_train": [], "rl_train": [], "val": [], "test": []
        }
        omitted_held_out = 0
        for s in samples:
            base_id = s.source_video_id
            s.split = source_split[base_id]
            splits[s.split].append(s)

            if (
                s.degradation_domain in {"synthetic_unseen", "natural"}
                and s.split != "test"
            ):
                splits[s.split].pop()
                s.split = ""
                omitted_held_out += 1

        for k, v in splits.items():
            logger.info("Split '%s': %d samples", k, len(v))
        if omitted_held_out:
            logger.info(
                "Excluded %d held-out-domain variants from non-test sources.",
                omitted_held_out,
            )
        return splits

    def assign_source_splits(
        self,
        clips: List[VideoClip],
        seed: Optional[int] = None,
    ) -> Dict[str, str]:
        """Freeze source assignments before constructing any profile variant."""
        split_seed = self.seed if seed is None else seed
        by_source = defaultdict(list)
        for clip in clips:
            by_source[clip.source_video_id].append(
                {
                    "source_dataset": clip.source_dataset,
                    "event_type": clip.event_type,
                }
            )
        outer = stratified_partition(
            by_source,
            fractions=(0.70, 0.15, 0.15),
            names=("train", "val", "test"),
            seed=split_seed,
        )
        training = {
            source_id: by_source[source_id]
            for source_id, split in outer.items()
            if split == "train"
        }
        inner = stratified_partition(
            training,
            fractions=(0.30, 0.70),
            names=("sft_train", "rl_train"),
            seed=split_seed + 1,
        )
        return {
            source_id: inner[source_id] if split == "train" else split
            for source_id, split in outer.items()
        }

    def validate_sample(self, sample: StructuredSample) -> bool:
        """Rule-based quality validation."""
        try:
            from model.parser import (
                extract_degradation_profile,
                parse_answer_fields,
            )

            if sample.difficulty_level not in SEVERITY_LEVELS:
                return False
            if sample.gt_interval[0] >= sample.gt_interval[1]:
                return False
            for block in [
                sample.type_annotation,
                sample.influence_annotation,
                sample.reasoning_annotation,
                sample.conclusion_annotation,
                sample.answer_annotation,
            ]:
                if not block or not block.strip():
                    return False
            if extract_degradation_profile(sample.type_annotation) != [
                (name, float(severity))
                for name, severity in sample.degradation_profile
            ]:
                return False
            answer = parse_answer_fields(
                sample.answer_annotation,
                event_active=bool(sample.task_mask["event"]),
                temporal_active=bool(sample.task_mask["temporal"]),
                duration_sec=sample.duration_sec,
            )
            if answer is None:
                return False
            if sample.task_mask["event"] and answer.event_type != sample.event_type:
                return False
            if sample.task_mask["temporal"] and answer.interval != tuple(sample.gt_interval):
                return False
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def build(
        self,
        source_dirs: List[str],
        annotation_file: str,
    ) -> Dict[str, List[StructuredSample]]:
        """Run the complete five-stage pipeline and return split samples."""
        clips = self.collect_and_segment(source_dirs, annotation_file)
        source_split = self.assign_source_splits(clips)
        splits: Dict[str, List[StructuredSample]] = {
            "sft_train": [], "rl_train": [], "val": [], "test": []
        }
        for clip in clips:
            profiles = self._build_profiles(clip)
            for profile in profiles:
                assigned_split = source_split[clip.source_video_id]
                if (
                    profile.domain in {"synthetic_unseen", "natural"}
                    and assigned_split != "test"
                ):
                    continue
                degraded = synthesize_degradation(clip, profile, seed=self.seed)
                degraded.video_id = f"{clip.video_id}__{self._profile_suffix(profile)}"
                sample = self._annotate(degraded, profile)
                if sample is None:
                    continue
                if not self.validate_sample(sample):
                    logger.warning("Sample %s failed validation, skipping.", sample.video_id)
                    continue
                sample.split = assigned_split
                splits[assigned_split].append(sample)

        return splits
