"""SurvVAU dataset builder — five-stage pipeline."""
from __future__ import annotations
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .annotation_pipeline import (
    adjust_compactness,
    compute_aggregated_difficulty,
    generate_answer,
    generate_influence,
    generate_reasoning,
)
from .augmentation import synthesize_difficulty
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

    # ------------------------------------------------------------------
    # Stage 1: collect & segment
    # ------------------------------------------------------------------

    def collect_and_segment(
        self,
        source_dirs: List[str],
        annotation_file: Optional[str] = None,
    ) -> List[VideoClip]:
        """Load source videos and wrap them as VideoClip objects.

        Args:
            source_dirs: Directories containing source video files.
            annotation_file: Optional JSON file with {video_id: {start_frame, end_frame,
                             start_sec, end_sec}} annotations.

        Returns:
            List of VideoClip objects.
        """
        annotations: Dict = {}
        if annotation_file:
            with open(annotation_file) as f:
                annotations = json.load(f)

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
                start_frame = ann.get("start_frame", 0)
                end_frame = ann.get("end_frame", len(frames) - 1)
                start_sec = ann.get("start_sec", 0.0)
                end_sec = ann.get("end_sec", len(frames) / 30.0)

                try:
                    clip = VideoClip(
                        video_id=video_id,
                        frames=frames,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        source_path=str(video_path),
                    )
                    clips.append(clip)
                except ValueError as e:
                    logger.error("Invalid clip %s: %s", video_id, e)

        logger.info("Collected %d clips from %d directories.", len(clips), len(source_dirs))
        return clips

    # ------------------------------------------------------------------
    # Stage 2: difficulty synthesis
    # ------------------------------------------------------------------

    def _build_profiles(self) -> List[DegradationProfile]:
        """Build one DegradationProfile per non-zero severity level."""
        profiles = []
        for level in SEVERITY_LEVELS:
            if level == 0.0:
                profiles.append(DegradationProfile(factors=[], difficulty_level=0.0))
            else:
                # Example: apply motion_blur + low_light at the given severity
                profiles.append(DegradationProfile(
                    factors=[("motion_blur", level), ("low_light", level)],
                    difficulty_level=level,
                ))
        return profiles

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
            answer = generate_answer(degraded, conclusion, self.model_q)

            # Stage 5: compactness adjustment
            s_bar = compute_aggregated_difficulty(profile)
            reasoning_adj = adjust_compactness(reasoning, s_bar, self.model_q)

            type_annotation = "; ".join(
                f"{name}:{sev:.1f}" for name, sev in profile.factors
            ) or "none"

            sample = StructuredSample(
                video_id=degraded.video_id,
                frames=degraded.frames,
                prompt=DEFAULT_PROMPT,
                degradation_profile=profile.factors,
                difficulty_level=profile.difficulty_level,
                gt_interval=(degraded.start_sec, degraded.end_sec),
                type_annotation=type_annotation,
                influence_annotation=influence,
                reasoning_annotation=reasoning_adj,
                conclusion_annotation=conclusion,
                answer_annotation=answer,
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

        # Group by base video_id (strip difficulty suffix if present)
        video_ids = list({s.video_id.split("_diff")[0] for s in samples})
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
            base_id = s.video_id.split("_diff")[0]
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
        annotation_file: Optional[str] = None,
    ) -> Dict[str, List[StructuredSample]]:
        """Run the complete five-stage pipeline and return split samples."""
        clips = self.collect_and_segment(source_dirs, annotation_file)
        profiles = self._build_profiles()

        all_samples: List[StructuredSample] = []
        for clip in clips:
            for profile in profiles:
                degraded = synthesize_difficulty(clip, profile)
                sample = self._annotate(degraded, profile)
                if sample is None:
                    continue
                if not self.validate_sample(sample):
                    logger.warning("Sample %s failed validation, skipping.", sample.video_id)
                    continue
                all_samples.append(sample)

        return self.split_dataset(all_samples)
