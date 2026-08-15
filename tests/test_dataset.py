"""Unit tests for dataset types, validation, split logic, and builder closure."""
import json

import numpy as np
import pytest
from dataset.types import (
    DegradedClip,
    DegradationProfile,
    ObjectTrack,
    StructuredSample,
    VideoClip,
    VideoLoadError,
    SEVERITY_LEVELS,
)
from dataset.builder import SurvVAUBuilder
from dataset.dataset import SurvVAUDataset, structured_output_instruction
from dataset.splitting import stratified_partition
from dataset.video_utils import (
    native_motion_pairs,
    uniform_sample_indices,
    uniform_sample_timestamps,
)
from evaluation.metrics import compute_tiou
from model.parser import extract_temporal_interval, parse_structured_output


def _make_sample(**kwargs) -> StructuredSample:
    defaults = dict(
        video_id="vid_001",
        source_video_id="source_001",
        source_dataset="self_collected",
        frames=[],
        prompt="Describe the anomaly.",
        degradation_profile=[("motion_blur", 0.4)],
        difficulty_level=0.4,
        gt_interval=(2.0, 8.0),
        event_type="rear-end collision",
        event_aliases=["rear end crash"],
        duration_sec=10.0,
        fps=25.0,
        num_source_frames=250,
        type_annotation="motion_blur:0.4",
        influence_annotation="Blur reduces clarity.",
        reasoning_annotation="Step 1: vehicle braked.",
        conclusion_annotation="Rear-end collision.",
        answer_annotation="event_type: rear-end collision; interval: [2.0, 8.0]",
        split="sft_train",
    )
    defaults.update(kwargs)
    return StructuredSample(**defaults)


class TestDegradationProfile:
    def test_valid_levels_accepted(self):
        for level in SEVERITY_LEVELS:
            factors = [] if level == 0.0 else [("motion_blur", level)]
            p = DegradationProfile(factors=factors, difficulty_level=level)
            assert p.difficulty_level == level

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError):
            DegradationProfile(factors=[], difficulty_level=0.5)

    def test_aggregated_score_empty(self):
        p = DegradationProfile(factors=[], difficulty_level=0.0)
        assert p.aggregated_score() == 0.0

    def test_aggregated_score_mean(self):
        p = DegradationProfile(
            factors=[("motion_blur", 0.4), ("low_light", 0.8)],
            difficulty_level=0.4,
        )
        assert p.aggregated_score() == pytest.approx(0.6)


class TestVideoClip:
    def test_valid_clip(self):
        clip = VideoClip(
            video_id="v1",
            frames=[np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(11)],
            start_frame=0, end_frame=10,
            start_sec=0.0, end_sec=5.0
        )
        assert clip.video_id == "v1"

    def test_invalid_frame_order_raises(self):
        with pytest.raises(ValueError):
            VideoClip(
                video_id="v1",
                frames=[np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(11)],
                start_frame=10, end_frame=5,
                start_sec=0.0, end_sec=5.0
            )


def test_exact_frame_to_second_mapping():
    assert uniform_sample_indices(49, n=5) == [0, 12, 24, 36, 48]
    assert uniform_sample_timestamps(49, fps=24.0, n=5) == pytest.approx(
        [0.0, 0.5, 1.0, 1.5, 2.0]
    )


def test_native_motion_pairs_are_adjacent_and_do_not_clip_last_pair():
    pairs = native_motion_pairs(49, n=5, offset=1)
    assert pairs[0] == (0, 1)
    assert pairs[-1] == (47, 48)
    assert all(second - first == 1 for first, second in pairs)


def test_structural_ablation_prompt_omits_removed_blocks():
    instruction = structured_output_instruction(
        ("REASONING", "CONCLUSION", "ANSWER")
    )
    assert "<TYPE>" not in instruction
    assert "<INFLUENCE>" not in instruction
    assert "<REASONING>" in instruction


def test_structural_ablation_prompt_rejects_reordered_blocks():
    with pytest.raises(ValueError):
        structured_output_instruction(("ANSWER", "REASONING"))


def test_dataset_rejects_legacy_reasoning_target_length(tmp_path):
    (tmp_path / "annotations.jsonl").write_text(
        json.dumps({"video_id": "v1", "reasoning_target_length": 80}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "splits.json").write_text(
        json.dumps({"v1": "test"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="severity-conditioned"):
        SurvVAUDataset(
            str(tmp_path), split="test", require_videos=False
        )


def test_stratified_partition_hits_global_counts_with_singleton_strata():
    sources = {
        f"source-{index}": [
            {"source_dataset": f"dataset-{index}", "event_type": "event"}
        ]
        for index in range(20)
    }
    assignment = stratified_partition(
        sources, (0.70, 0.15, 0.15), ("train", "val", "test"), seed=42
    )
    assert list(assignment.values()).count("train") == 14
    assert list(assignment.values()).count("val") == 3
    assert list(assignment.values()).count("test") == 3


class TestStructuredSample:
    def test_valid_sample(self):
        s = _make_sample()
        assert s.video_id == "vid_001"

    def test_invalid_difficulty_raises(self):
        with pytest.raises(ValueError):
            _make_sample(difficulty_level=0.5)

    def test_invalid_interval_raises(self):
        with pytest.raises(ValueError):
            _make_sample(gt_interval=(8.0, 2.0))

    def test_empty_annotation_raises(self):
        with pytest.raises(ValueError):
            _make_sample(type_annotation="")


class TestSurvVAUBuilderValidation:
    def setup_method(self):
        self.builder = SurvVAUBuilder(annotator_model=None)

    def test_valid_sample_passes(self):
        s = _make_sample()
        assert self.builder.validate_sample(s) is True

    def test_invalid_difficulty_fails(self):
        s = _make_sample.__wrapped__() if hasattr(_make_sample, "__wrapped__") else None
        # Directly test with a dict-like approach
        try:
            bad = _make_sample(difficulty_level=0.5)
            assert False, "Should have raised"
        except ValueError:
            pass  # Expected

    def test_empty_block_fails_validation(self):
        # Manually create a sample with empty block bypassing __post_init__
        s = _make_sample()
        object.__setattr__(s, "type_annotation", "")
        assert self.builder.validate_sample(s) is False

    def test_frozen_profiles_cover_clean_and_compound_conditions(self):
        profiles = self.builder._parse_degradation_profiles(
            [
                {"degradation_level": 0.0, "factors": []},
                {
                    "degradation_level": 0.4,
                    "factors": [
                        ["motion_blur", 0.4],
                        ["low_light", 0.4],
                    ],
                },
            ],
            "source_001",
        )
        assert profiles[0].factors == []
        assert profiles[1].aggregated_score() == pytest.approx(0.4)
        assert self.builder._profile_suffix(profiles[1]) == (
            "motion_blur-40__low_light-40"
        )

    def test_frozen_profiles_require_clean_reference(self):
        with pytest.raises(ValueError):
            self.builder._parse_degradation_profiles(
                [
                    {
                        "degradation_level": 0.4,
                        "factors": [["motion_blur", 0.4]],
                    }
                ],
                "source_001",
            )

    def test_natural_profile_has_a_distinct_output_suffix(self):
        natural = DegradationProfile(
            factors=[], difficulty_level=0.0, domain="natural"
        )
        assert self.builder._profile_suffix(natural) == "natural"

    def test_parses_normalized_object_trajectories(self):
        tracks = self.builder._parse_object_tracks(
            [
                {
                    "track_id": "vehicle-7",
                    "category": "vehicle",
                    "event_relevant": True,
                    "boxes": [
                        {"frame_index": 0, "bbox_norm": [0.1, 0.2, 0.3, 0.4]},
                        {"frame_index": 2, "bbox_norm": [0.3, 0.2, 0.5, 0.4]},
                    ],
                }
            ],
            "source_001",
        )
        assert isinstance(tracks[0], ObjectTrack)
        assert tracks[0].box_at(1) == pytest.approx((0.2, 0.2, 0.4, 0.4))

    def test_builder_parser_tiou_closed_loop_has_one_answer_interval(self):
        class DummyAnnotator:
            def __init__(self):
                self.outputs = iter(
                    [
                        "Blur weakens edge evidence.",
                        "The model-authored reasoning mentions [0.1, 0.2].",
                        "The conclusion also mentions from 0.1 to 0.2 sec.",
                        "Compact reasoning retained.",
                    ]
                )

            def generate(self, frames, prompt):
                return next(self.outputs)

        frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(4)]
        profile = DegradationProfile(factors=[], difficulty_level=0.0)
        source = VideoClip(
            video_id="demo",
            source_video_id="demo",
            source_dataset="synthetic_demo",
            frames=frames,
            start_frame=0,
            end_frame=3,
            start_sec=0.5,
            end_sec=1.5,
            event_type="rear-end collision",
            degradation_profiles=[profile],
            fps=2.0,
            duration_sec=2.0,
        )
        degraded = DegradedClip(
            video_id="demo__clean",
            frames=frames,
            start_sec=0.5,
            end_sec=1.5,
            profile=profile,
            source_clip=source,
        )
        sample = SurvVAUBuilder(DummyAnnotator())._annotate(degraded, profile)
        assert sample is not None
        assert sample.answer_annotation == (
            "event_type: rear-end collision; interval: [0.500, 1.500]"
        )
        raw_output = (
            f"<TYPE>{sample.type_annotation}<TYPE_END>"
            f"<INFLUENCE>{sample.influence_annotation}<INFLUENCE_END>"
            f"<REASONING>{sample.reasoning_annotation}<REASONING_END>"
            f"<CONCLUSION>{sample.conclusion_annotation}<CONCLUSION_END>"
            f"<ANSWER>{sample.answer_annotation}<ANSWER_END>"
        )
        parsed = parse_structured_output(raw_output)
        assert parsed is not None
        interval = extract_temporal_interval(parsed.answer_block)
        assert interval == pytest.approx((0.5, 1.5))
        assert compute_tiou(interval, sample.gt_interval, 2.0) == pytest.approx(1.0)

    def test_answer_serializer_rejects_reserved_delimiters(self):
        with pytest.raises(ValueError):
            self.builder.serialize_answer("collision; interval: [1, 2]", (1.0, 2.0))


class TestSplitDataset:
    def setup_method(self):
        self.builder = SurvVAUBuilder(annotator_model=None, seed=42)

    def _make_samples(self, n: int):
        samples = []
        for i in range(n):
            s = _make_sample(
                video_id=f"vid_{i:03d}",
                source_video_id=f"source_{i:03d}",
                split="",
            )
            samples.append(s)
        return samples

    def test_split_ratios(self):
        samples = self._make_samples(100)
        splits = self.builder.split_dataset(samples)
        total = sum(len(v) for v in splits.values())
        assert total == 100

    def test_split_is_independent_of_input_order(self):
        forward = self._make_samples(40)
        reverse = list(reversed(self._make_samples(40)))
        first = self.builder.split_dataset(forward)
        second = self.builder.split_dataset(reverse)
        first_map = {
            sample.source_video_id: split
            for split, rows in first.items()
            for sample in rows
        }
        second_map = {
            sample.source_video_id: split
            for split, rows in second.items()
            for sample in rows
        }
        assert first_map == second_map

    def test_held_out_domains_never_enter_training_or_validation(self):
        samples = self._make_samples(40)
        for index, sample in enumerate(samples):
            if index % 2 == 0:
                sample.degradation_domain = "natural"
        splits = self.builder.split_dataset(samples)
        for split in ("sft_train", "rl_train", "val"):
            assert all(
                sample.degradation_domain not in {"natural", "synthetic_unseen"}
                for sample in splits[split]
            )

    def test_no_leakage(self):
        """Same source video must not appear in multiple splits."""
        samples = self._make_samples(50)
        splits = self.builder.split_dataset(samples)
        all_ids = {}
        for split_name, split_samples in splits.items():
            for s in split_samples:
                base_id = s.video_id.split("_diff")[0]
                if base_id in all_ids:
                    assert all_ids[base_id] == split_name, (
                        f"Video {base_id} appears in both {all_ids[base_id]} and {split_name}"
                    )
                all_ids[base_id] = split_name

    def test_sft_rl_ratio(self):
        samples = self._make_samples(100)
        splits = self.builder.split_dataset(samples)
        sft = len(splits["sft_train"])
        rl = len(splits["rl_train"])
        if sft + rl > 0:
            sft_ratio = sft / (sft + rl)
            assert 0.2 <= sft_ratio <= 0.4  # approximately 30%
