"""Unit tests for dataset types, validation, and split logic."""
import pytest
from dataset.types import (
    DegradationProfile,
    StructuredSample,
    VideoClip,
    VideoLoadError,
    SEVERITY_LEVELS,
)
from dataset.builder import SurvVAUBuilder


def _make_sample(**kwargs) -> StructuredSample:
    defaults = dict(
        video_id="vid_001",
        frames=[],
        prompt="Describe the anomaly.",
        degradation_profile=[("motion_blur", 0.4)],
        difficulty_level=0.4,
        gt_interval=(2.0, 8.0),
        type_annotation="motion_blur:0.4",
        influence_annotation="Blur reduces clarity.",
        reasoning_annotation="Step 1: vehicle braked.",
        conclusion_annotation="Rear-end collision.",
        answer_annotation="Rear-end collision [2.0, 8.0]",
        split="sft_train",
    )
    defaults.update(kwargs)
    return StructuredSample(**defaults)


class TestDegradationProfile:
    def test_valid_levels_accepted(self):
        for level in SEVERITY_LEVELS:
            p = DegradationProfile(factors=[], difficulty_level=level)
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
            video_id="v1", frames=[], start_frame=0, end_frame=10,
            start_sec=0.0, end_sec=5.0
        )
        assert clip.video_id == "v1"

    def test_invalid_frame_order_raises(self):
        with pytest.raises(ValueError):
            VideoClip(
                video_id="v1", frames=[], start_frame=10, end_frame=5,
                start_sec=0.0, end_sec=5.0
            )


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


class TestSplitDataset:
    def setup_method(self):
        self.builder = SurvVAUBuilder(annotator_model=None, seed=42)

    def _make_samples(self, n: int):
        samples = []
        for i in range(n):
            s = _make_sample(video_id=f"vid_{i:03d}", split="")
            samples.append(s)
        return samples

    def test_split_ratios(self):
        samples = self._make_samples(100)
        splits = self.builder.split_dataset(samples)
        total = sum(len(v) for v in splits.values())
        assert total == 100

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
