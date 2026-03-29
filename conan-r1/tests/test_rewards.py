"""Unit tests for reward functions."""
import pytest
from training.rewards import (
    compute_ro,
    compute_rt,
    compute_rl,
    compute_total_reward,
    effective_length,
)


class TestComputeRo:
    def test_perfect_match_returns_one(self):
        profile = [("motion_blur", 0.4), ("low_light", 0.2)]
        assert compute_ro(profile, profile) == pytest.approx(1.0, abs=1e-6)

    def test_empty_both_returns_one(self):
        assert compute_ro([], []) == pytest.approx(1.0)

    def test_all_false_positives(self):
        pred = [("motion_blur", 0.4)]
        gt = []
        ro = compute_ro(pred, gt, lambda_fp=0.3)
        assert ro == pytest.approx(-0.3)

    def test_all_false_negatives(self):
        pred = []
        gt = [("motion_blur", 0.4)]
        ro = compute_ro(pred, gt, lambda_fn=0.3)
        assert ro == pytest.approx(-0.3)

    def test_ro_upper_bound(self):
        pred = [("fog", 0.8)]
        gt = [("fog", 0.8)]
        ro = compute_ro(pred, gt)
        assert ro <= 1.0

    def test_severity_deviation_penalized(self):
        pred = [("fog", 0.0)]
        gt = [("fog", 0.8)]
        ro_perfect = compute_ro([("fog", 0.8)], [("fog", 0.8)])
        ro_deviated = compute_ro(pred, gt)
        assert ro_deviated < ro_perfect


class TestComputeRt:
    def test_perfect_overlap(self):
        assert compute_rt((2.0, 8.0), (2.0, 8.0)) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert compute_rt((0.0, 3.0), (5.0, 10.0)) == pytest.approx(0.0)

    def test_none_interval(self):
        assert compute_rt(None, (2.0, 8.0)) == pytest.approx(0.0)

    def test_partial_overlap(self):
        rt = compute_rt((0.0, 6.0), (4.0, 10.0))
        assert 0.0 < rt < 1.0

    def test_bounds(self):
        rt = compute_rt((1.0, 5.0), (3.0, 9.0))
        assert 0.0 <= rt <= 1.0


class TestComputeRl:
    def test_identical_reasoning(self):
        text = "The vehicle braked suddenly causing a rear-end collision."
        assert compute_rl(text, text) == pytest.approx(1.0)

    def test_empty_pred(self):
        rl = compute_rl("", "some reasoning text here")
        assert 0.0 <= rl <= 1.0

    def test_bounds(self):
        rl = compute_rl("short text", "a much longer reasoning chain with many details")
        assert 0.0 <= rl <= 1.0


class TestEffectiveLength:
    def test_shorter_than_original(self):
        text = "the car the car the car braked suddenly"
        assert effective_length(text) <= len(text.split())

    def test_empty_string(self):
        assert effective_length("") == 0

    def test_no_repetition(self):
        text = "vehicle braked suddenly causing collision"
        result = effective_length(text)
        assert result <= len(text.split())


class TestComputeTotalReward:
    def test_weights_sum_to_one(self):
        alpha, beta, gamma = 0.4, 0.4, 0.2
        assert alpha + beta + gamma == pytest.approx(1.0)

    def test_all_ones(self):
        assert compute_total_reward(1.0, 1.0, 1.0) == pytest.approx(1.0)

    def test_all_zeros(self):
        assert compute_total_reward(0.0, 0.0, 0.0) == pytest.approx(0.0)

    def test_weighted_combination(self):
        result = compute_total_reward(1.0, 0.0, 0.0, alpha=0.4, beta=0.4, gamma=0.2)
        assert result == pytest.approx(0.4)
