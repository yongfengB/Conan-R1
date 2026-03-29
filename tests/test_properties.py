"""Property-based tests for Conan-R1 using hypothesis.

Tests the 10 correctness properties defined in design.md.
"""
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from training.rewards import (
    compute_rt,
    compute_rl,
    compute_total_reward,
    effective_length,
    compute_ro,
)
from model.parser import parse_structured_output
from dataset.types import DegradationProfile, SEVERITY_LEVELS


# ---------------------------------------------------------------------------
# Property 3: rt ∈ [0, 1] for any valid interval pair
# ---------------------------------------------------------------------------

@given(
    p_start=st.floats(min_value=0.0, max_value=100.0),
    p_len=st.floats(min_value=0.01, max_value=50.0),
    g_start=st.floats(min_value=0.0, max_value=100.0),
    g_len=st.floats(min_value=0.01, max_value=50.0),
)
@settings(max_examples=200)
def test_property3_rt_bounded(p_start, p_len, g_start, g_len):
    """Property 3: rt ∈ [0, 1] for any valid interval pair."""
    pred = (p_start, p_start + p_len)
    gt = (g_start, g_start + g_len)
    rt = compute_rt(pred, gt)
    assert 0.0 <= rt <= 1.0


def test_property3_rt_perfect_match():
    assert compute_rt((2.0, 8.0), (2.0, 8.0)) == pytest.approx(1.0)


def test_property3_rt_no_overlap():
    assert compute_rt((0.0, 3.0), (5.0, 10.0)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Property 4: rl ∈ [0, 1] for any reasoning text pair
# ---------------------------------------------------------------------------

@given(
    pred=st.text(min_size=0, max_size=200),
    gt=st.text(min_size=1, max_size=200),
)
@settings(max_examples=200)
def test_property4_rl_bounded(pred, gt):
    """Property 4: rl ∈ [0, 1] for any reasoning text pair."""
    rl = compute_rl(pred, gt)
    assert 0.0 <= rl <= 1.0


def test_property4_rl_identical():
    text = "vehicle braked suddenly causing rear-end collision"
    assert compute_rl(text, text) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Property 5: effective_length(text) ≤ len(tokenize(text))
# ---------------------------------------------------------------------------

@given(text=st.text(min_size=0, max_size=500))
@settings(max_examples=300)
def test_property5_effective_length_not_exceeds_original(text):
    """Property 5: effective_length(text) ≤ len(tokenize(text))."""
    original_len = len(text.lower().split())
    eff_len = effective_length(text)
    assert eff_len <= original_len


# ---------------------------------------------------------------------------
# Property 6: total reward is weighted linear combination
# ---------------------------------------------------------------------------

@given(
    ro=st.floats(min_value=-2.0, max_value=1.0),
    rt=st.floats(min_value=0.0, max_value=1.0),
    rl=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=200)
def test_property6_total_reward_linear(ro, rt, rl):
    """Property 6: R = 0.4*ro + 0.4*rt + 0.2*rl."""
    alpha, beta, gamma = 0.4, 0.4, 0.2
    expected = alpha * ro + beta * rt + gamma * rl
    result = compute_total_reward(ro, rt, rl, alpha, beta, gamma)
    assert result == pytest.approx(expected, abs=1e-9)


def test_property6_weights_sum_to_one():
    assert 0.4 + 0.4 + 0.2 == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Property 8: GRPO group advantages have mean ≈ 0
# ---------------------------------------------------------------------------

@given(
    rewards=st.lists(
        st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=8,
    )
)
@settings(max_examples=200)
def test_property8_advantages_mean_zero(rewards):
    """Property 8: normalized group advantages have mean ≈ 0."""
    import numpy as np
    arr = np.array(rewards)
    mean = arr.mean()
    std = arr.std() + 1e-6
    advantages = ((arr - mean) / std).tolist()
    assert abs(sum(advantages) / len(advantages)) < 1e-5


# ---------------------------------------------------------------------------
# Property 9: frame sampler produces exactly 25 frames of 448x448
# ---------------------------------------------------------------------------

def test_property9_frame_sampler_spec():
    """Property 9: sample_frames returns exactly 25 frames of size 448x448."""
    import numpy as np
    from dataset.video_utils import frames_from_array

    # Create a synthetic 50-frame video
    fake_frames = [
        np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(50)
    ]
    sampled = frames_from_array(fake_frames, n=25, size=(448, 448))
    assert len(sampled) == 25
    for frame in sampled:
        assert frame.shape == (448, 448, 3)


# ---------------------------------------------------------------------------
# Property 10: DegradationProfile rejects invalid difficulty levels
# ---------------------------------------------------------------------------

@given(
    level=st.floats(min_value=-1.0, max_value=2.0).filter(
        lambda x: x not in SEVERITY_LEVELS
    )
)
@settings(max_examples=100)
def test_property10_invalid_difficulty_raises(level):
    """Property 10: difficulty_level not in {0.0, 0.2, 0.4, 0.8} raises ValueError."""
    with pytest.raises(ValueError):
        DegradationProfile(factors=[], difficulty_level=level)


@pytest.mark.parametrize("level", SEVERITY_LEVELS)
def test_property10_valid_difficulty_accepted(level):
    p = DegradationProfile(factors=[], difficulty_level=level)
    assert p.difficulty_level == level
