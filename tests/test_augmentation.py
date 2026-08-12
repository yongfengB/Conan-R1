"""Tests for object-aware and temporally coherent degradation synthesis."""
import numpy as np
import pytest

from dataset.augmentation import OPERATOR_ORDER, synthesize_degradation
from dataset.degradation_protocol import (
    load_degradation_protocol,
    sample_degradation_profile,
)
from dataset.types import (
    DegradationProfile,
    ObjectTrack,
    SpatialAnnotationError,
    TrackBox,
    VideoClip,
)


def _track(track_id, boxes, category="vehicle"):
    return ObjectTrack(
        track_id=track_id,
        category=category,
        boxes=[TrackBox(frame_index=index, bbox=box) for index, box in boxes],
    )


def _clip(frames, tracks=None):
    return VideoClip(
        video_id="clip-1",
        source_video_id="source-1",
        frames=frames,
        start_frame=0,
        end_frame=len(frames) - 1,
        start_sec=0.0,
        end_sec=1.0,
        duration_sec=1.0,
        fps=float(max(1, len(frames) - 1)),
        object_tracks=tracks or [],
    )


def test_vehicle_mask_follows_interpolated_trajectory():
    frames = [np.full((100, 100, 3), 255, dtype=np.uint8) for _ in range(3)]
    vehicle = _track(
        "vehicle-1",
        [(0, (0.10, 0.40, 0.30, 0.60)), (2, (0.70, 0.40, 0.90, 0.60))],
    )
    result = synthesize_degradation(
        _clip(frames, [vehicle]),
        DegradationProfile([("vehicle_mask", 0.8)], 0.8),
        seed=7,
    )
    assert result.frames[0][50, 20].mean() < 80
    assert result.frames[0][50, 80].mean() > 240
    assert result.frames[2][50, 80].mean() < 80
    assert result.frames[2][50, 20].mean() > 240


def test_vehicle_mask_never_falls_back_to_a_center_rectangle():
    frames = [np.full((32, 32, 3), 255, dtype=np.uint8) for _ in range(3)]
    with pytest.raises(SpatialAnnotationError):
        synthesize_degradation(
            _clip(frames),
            DegradationProfile([("vehicle_mask", 0.4)], 0.4),
        )


def test_interaction_mask_uses_a_stable_track_pair_not_top_left():
    frames = [np.full((100, 100, 3), 255, dtype=np.uint8) for _ in range(3)]
    first = _track("a", [(0, (0.30, 0.40, 0.42, 0.62)), (2, (0.36, 0.40, 0.48, 0.62))])
    second = _track("b", [(0, (0.50, 0.40, 0.62, 0.62)), (2, (0.44, 0.40, 0.56, 0.62))])
    result = synthesize_degradation(
        _clip(frames, [first, second]),
        DegradationProfile([("interaction_area_mask", 0.8)], 0.8),
    )
    assert result.frames[1][50, 45].mean() < 80
    assert result.frames[1][5, 5].mean() > 240


def test_video_level_random_state_is_deterministic_and_temporally_correlated():
    frames = [np.full((48, 48, 3), 128, dtype=np.uint8) for _ in range(5)]
    profile = DegradationProfile([("sensor_noise", 0.2)], 0.2)
    first = synthesize_degradation(_clip(frames), profile, seed=11)
    second = synthesize_degradation(_clip(frames), profile, seed=11)
    for left, right in zip(first.frames, second.frames):
        assert np.array_equal(left, right)
    noise0 = first.frames[0].astype(float) - 128.0
    noise1 = first.frames[1].astype(float) - 128.0
    correlation = np.corrcoef(noise0.ravel(), noise1.ravel())[0, 1]
    assert correlation > 0.70


def test_persistent_weather_changes_smoothly_but_is_not_static():
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(4)]
    profile = DegradationProfile([("rain_snow", 0.4)], 0.4)
    result = synthesize_degradation(_clip(frames), profile, seed=19)
    assert not np.array_equal(result.frames[0], result.frames[1])
    repeat = synthesize_degradation(_clip(frames), profile, seed=19)
    assert np.array_equal(result.frames[1], repeat.frames[1])


def test_protocol_defines_and_samples_k_distribution():
    protocol = load_degradation_protocol()
    assert OPERATOR_ORDER == protocol["operator_order"]
    rng = np.random.default_rng(42)
    counts = {1: 0, 2: 0, 3: 0}
    for _ in range(4000):
        profile = sample_degradation_profile(rng, 0.4, protocol)
        counts[len(profile.factors)] += 1
    frequencies = {key: value / 4000.0 for key, value in counts.items()}
    assert frequencies[1] == pytest.approx(0.60, abs=0.04)
    assert frequencies[2] == pytest.approx(0.30, abs=0.04)
    assert frequencies[3] == pytest.approx(0.10, abs=0.03)


def test_unseen_profile_contains_a_held_out_operator():
    protocol = load_degradation_protocol()
    profile = sample_degradation_profile(
        np.random.default_rng(3), 0.8, protocol, domain="synthetic_unseen"
    )
    names = {name for name, _ in profile.factors}
    assert names & set(protocol["synthetic_unseen_test_operators"])


def test_declared_unseen_combinations_are_not_training_reachable():
    protocol = load_degradation_protocol()
    categories = protocol["seen_training_operators"]
    category_by_operator = {
        operator: category
        for category, operators in categories.items()
        for operator in operators
    }
    for combination in protocol["synthetic_unseen_test_combinations"]:
        represented = [category_by_operator[name] for name in combination]
        assert len(set(represented)) < len(represented)


def test_synthesis_metadata_records_exact_operator_contract():
    frames = [np.full((32, 32, 3), 128, dtype=np.uint8) for _ in range(3)]
    profile = DegradationProfile(
        [("sensor_noise", 0.4), ("motion_blur", 0.4)], 0.4
    )
    result = synthesize_degradation(_clip(frames), profile, seed=5)
    metadata = result.synthesis_metadata
    assert metadata["synthesis_applied"] is True
    assert metadata["operator_order"] == ["motion_blur", "sensor_noise"]
    assert [item["severity_fraction"] for item in metadata["active_operators"]] == [
        0.4,
        0.4,
    ]
    assert all(item["maximum_magnitude"] for item in metadata["active_operators"])
