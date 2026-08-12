"""Tests that distinguish a real clipped update from the old ratio=1 bug."""
import pytest
import torch

from training.grpo_math import (
    clipped_grpo_loss,
    normalize_group_advantages,
    sampled_reverse_kl,
)


def test_advantages_are_zero_for_equal_rewards():
    advantages = normalize_group_advantages(torch.tensor([0.5, 0.5, 0.5, 0.5]))
    assert torch.equal(advantages, torch.zeros_like(advantages))


def test_advantages_have_zero_mean():
    advantages = normalize_group_advantages(torch.tensor([0.1, 0.4, 0.8, 0.2]))
    assert float(advantages.mean()) == pytest.approx(0.0, abs=1e-6)


def test_clipping_activates_when_current_policy_moves():
    old = torch.tensor([-2.0, -2.0])
    current = torch.tensor([-1.0, -3.0], requires_grad=True)
    reference = torch.tensor([-2.0, -2.0])
    loss, diagnostics = clipped_grpo_loss(
        current, old, reference, torch.tensor(1.0), clip_eps=0.2, kl_coef=0.02
    )
    loss.backward()
    assert float(diagnostics["ratio_mean"]) != pytest.approx(1.0)
    assert float(diagnostics["clip_fraction"]) > 0.0
    assert current.grad is not None


def test_sampled_kl_is_non_negative():
    current = torch.tensor([-1.0, -2.0])
    reference = torch.tensor([-1.5, -1.5])
    assert torch.all(sampled_reverse_kl(current, reference) >= 0.0)


def test_sampled_kl_remains_finite_for_extreme_log_ratios():
    current = torch.tensor([-1000.0, 1000.0], requires_grad=True)
    reference = torch.tensor([1000.0, -1000.0])
    kl = sampled_reverse_kl(current, reference)
    assert torch.all(torch.isfinite(kl))
    kl.mean().backward()
    assert torch.all(torch.isfinite(current.grad))


@pytest.mark.parametrize(
    "clip_eps,kl_coef",
    [(0.0, 0.1), (1.0, 0.1), (0.2, -0.1)],
)
def test_invalid_grpo_hyperparameters_are_rejected(clip_eps, kl_coef):
    values = torch.tensor([-1.0])
    with pytest.raises(ValueError):
        clipped_grpo_loss(
            values, values, values, torch.tensor(1.0), clip_eps, kl_coef
        )
