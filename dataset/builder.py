"""SurvVAU dataset builder — five-stage pipeline."""
from __future__ import annotations
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .annotation_pipeline import (
    adjust_compactness,
    compute_aggregated_severity,
    derive_reasoning_target_length,
    generate_answer,
    generate_influence,
    generate_reasoning,
)
from .augmentation import synthesize_degradation
from .types import (
    SEVERITY_LEVELS,
    DegradationProfile,
    DegradedClip,
    StructuredSample,
    VideoClip,
    VideoLoadError,
)
from .video_utils import frames_from_array, load_video

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
                DegradationProfile(factors=factors, difficulty_level=level)
            )
        if not any(not profile.factors for profile in profiles):
            raise ValueError(f"{video_id}: a clean 0% profile is required")
        return profiles

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
                        event_type=event_type,
                        event_aliases=list(ann.get("event_aliases", [])),
                        degradation_profiles=degradation_profiles,
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
        return clip.degradation_profiles

    @staticmethod
    def _profile_suffix(profile: DegradationProfile) -> str:
        if not profile.factors:
            return "clean"
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
            generated_explanation = generate_answer(
                degraded, conclusion, self.model_q
            )

            # Stage 5: compactness adjustment
            s_bar = compute_aggregated_severity(profile)
            reasoning_adj = adjust_compactness(reasoning, s_bar, self.model_q)
            reasoning_target_length = derive_reasoning_target_length(s_bar)

            type_annotation = "; ".join(
                f"{name}:{sev:.1f}" for name, sev in profile.factors
            ) or "none"

            sample = StructuredSample(
                video_id=degraded.video_id,
                source_video_id=degraded.source_clip.source_video_id,
                source_dataset=degraded.source_clip.source_dataset,
                frames=degraded.frames,
                prompt=DEFAULT_PROMPT,
                degradation_profile=profile.factors,
                difficulty_level=profile.difficulty_level,
                gt_interval=(degraded.start_sec, degraded.end_sec),
                event_type=degraded.source_clip.event_type,
                event_aliases=degraded.source_clip.event_aliases,
                reasoning_target_length=reasoning_target_length,
                reasoning_target_source="deterministic_policy",
                duration_sec=degraded.source_clip.duration_sec,
                fps=degraded.source_clip.fps,
                num_source_frames=len(degraded.source_clip.frames),
                type_annotation=type_annotation,
                influence_annotation=influence,
                reasoning_annotation=reasoning_adj,
                conclusion_annotation=conclusion,
                answer_annotation=(
                    f"event_type: {degraded.source_clip.event_type}; "
                    f"interval: [{degraded.start_sec:.3f}, "
                    f"{degraded.end_sec:.3f}]; explanation: "
                    f"{generated_explanation.strip()}"
                ),
                split="",  # assigned later
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
        rng = random.Random(seed if seed is not None else self.seed)

        # Group by immutable source_video_id.
        video_ids = list({s.source_video_id for s in samples})
        rng.shuffle(video_ids)

        n = len(video_ids)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)

        train_ids = set(video_ids[:n_train])
        val_ids = set(video_ids[n_train: n_train + n_val])
        test_ids = set(video_ids[n_train + n_val:])

        n_sft = int(len(train_ids) * 0.30)
        train_list = list(train_ids)
        rng.shuffle(train_list)
        sft_ids = set(train_list[:n_sft])
        rl_ids = set(train_list[n_sft:])

        splits: Dict[str, List[StructuredSample]] = {
            "sft_train": [], "rl_train": [], "val": [], "test": []
        }
        for s in samples:
            base_id = s.source_video_id
            if base_id in sft_ids:
                s.split = "sft_train"
                splits["sft_train"].append(s)
            elif base_id in rl_ids:
                s.split = "rl_train"
                splits["rl_train"].append(s)
            elif base_id in val_ids:
                s.split = "val"
                splits["val"].append(s)
            else:
                s.split = "test"
                splits["test"].append(s)

        for k, v in splits.items():
            logger.info("Split '%s': %d samples", k, len(v))
        return splits

    def validate_sample(self, sample: StructuredSample) -> bool:
        """Rule-based quality validation."""
        try:
            # Re-run __post_init__ checks via a dummy re-assignment
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
        all_samples: List[StructuredSample] = []
        for clip in clips:
            profiles = self._build_profiles(clip)
            for profile in profiles:
                degraded = synthesize_degradation(clip, profile)
                degraded.video_id = f"{clip.video_id}__{self._profile_suffix(profile)}"
                sample = self._annotate(degraded, profile)
                if sample is None:
                    continue
                if not self.validate_sample(sample):
                    logger.warning("Sample %s failed validation, skipping.", sample.video_id)
                    continue
                all_samples.append(sample)

        return self.split_dataset(all_samples)
