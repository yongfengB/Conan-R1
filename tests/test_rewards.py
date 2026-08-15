"""Unit tests for the four verifiable rewards."""
import math

import pytest

from training.rewards import (
    compute_rd,
    compute_re,
    compute_rl,
    compute_ro,
    compute_rt,
    compute_total_reward,
    compute_task_masked_reward,
    compactness_budget,
    effective_length,
    validate_reward_weights,
)


def test_deprecated_reward_alias_matches():
    pred = [("motion blur", 0.4)]
    gt = [("motion_blur", 0.2)]
    assert compute_rd(pred, gt) == pytest.approx(compute_ro(pred, gt))


@pytest.mark.parametrize(
    "pred,gt,expected",
    [
        ([], [], 1.0),
        ([("fog", 0.8)], [("fog", 0.8)], 1.0),
        ([("fog", 0.8)], [], 0.0),
        ([], [("fog", 0.8)], 0.0),
    ],
)
def test_rd_edge_cases(pred, gt, expected):
    assert compute_rd(pred, gt) == pytest.approx(expected)


def test_rd_is_bounded_and_penalizes_errors():
    perfect = compute_rd([("fog", 0.8)], [("fog", 0.8)])
    wrong = compute_rd(
        [("fog", 0.0), ("rain_snow", 0.4)],
        [("fog", 0.8), ("low_light", 0.4)],
    )
    assert 0.0 <= wrong < perfect <= 1.0


def test_event_reward_uses_manifest_aliases_only():
    assert compute_re("Rear-End Collision", "rear end collision") == 1.0
    assert compute_re("crash", "rear end collision", aliases=["crash"]) == 1.0
    assert compute_re("lane departure", "rear end collision") == 0.0
    assert compute_re(None, "rear end collision") == 0.0


def test_temporal_reward_validates_intervals():
    assert compute_rt((2.0, 8.0), (2.0, 8.0)) == 1.0
    assert compute_rt((8.0, 2.0), (2.0, 8.0)) == 0.0
    assert compute_rt(None, (2.0, 8.0)) == 0.0
    assert compute_rt((2.0, 12.0), (2.0, 8.0), duration_sec=10.0) == 0.0
    assert 0.0 < compute_rt((0.0, 6.0), (4.0, 10.0)) < 1.0


def test_length_reward_uses_one_sided_task_budget():
    text = "vehicle brakes and the following vehicle collides"
    assert compute_rl(text, event_active=True, temporal_active=False) == 1.0
    long_text = " ".join(f"token{i}" for i in range(120))
    assert 0.0 < compute_rl(long_text) < 1.0
    assert compute_rl("", event_active=True, temporal_active=False) == 1.0
    with pytest.raises(TypeError):
        compute_rl(text, base_budget="64")


def test_length_reward_matches_paper_eq12_boundaries():
    single_64 = " ".join(f"s{i}" for i in range(64))
    single_65 = single_64 + " overflow"
    joint_96 = " ".join(f"j{i}" for i in range(96))
    joint_97 = joint_96 + " overflow"

    assert compactness_budget(True, False) == 64
    assert compactness_budget(False, True) == 64
    assert compactness_budget(True, True) == 96
    assert compute_rl(single_64, temporal_active=False) == 1.0
    assert compute_rl(single_65, temporal_active=False) == pytest.approx(
        math.exp(-1.0 / 64.0)
    )
    assert compute_rl(joint_96) == 1.0
    assert compute_rl(joint_97) == pytest.approx(math.exp(-1.0 / 96.0))


def test_total_reward_and_weight_validation():
    assert compute_total_reward(1.0, 1.0, 1.0, 1.0) == 1.0
    assert compute_total_reward(0.0, 0.0, 0.0, 0.0) == 0.0
    assert compute_total_reward(
        1.0, 0.0, 0.0, 0.0, w_d=0.4, w_e=0.2, w_t=0.2, w_l=0.2
    ) == pytest.approx(0.4)
    with pytest.raises(ValueError):
        validate_reward_weights(
            {"w_d": 0.4, "w_e": 0.4, "w_t": 0.4, "w_l": 0.0}
        )


def test_task_masked_reward_renormalizes_active_weights():
    score = compute_task_masked_reward(
        1.0,
        0.0,
        0.0,
        1.0,
        event_active=False,
        temporal_active=False,
    )
    assert score == pytest.approx(1.0)
    score = compute_task_masked_reward(
        1.0,
        1.0,
        0.0,
        1.0,
        event_active=True,
        temporal_active=False,
    )
    assert score == pytest.approx(1.0)


def test_malformed_active_field_zeroes_all_active_rewards():
    assert compute_task_masked_reward(
        1.0,
        1.0,
        1.0,
        1.0,
        event_active=True,
        temporal_active=True,
        active_fields_valid=False,
    ) == 0.0
