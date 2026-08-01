"""Pure tensor operations for the clipped GRPO objective."""
from __future__ import annotations

from typing import Dict, Tuple

import torch


def normalize_group_advantages(
    rewards: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    """Normalize rewards within a sampled group.

    Equal-reward groups receive zero advantages instead of numerical noise.
    """
    rewards = rewards.float()
    if rewards.numel() < 2:
        raise ValueError("GRPO requires at least two candidates per group.")
    standard_deviation = rewards.std(unbiased=False)
    if standard_deviation <= epsilon:
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / (standard_deviation + epsilon)


def sampled_reverse_kl(
    current_log_probs: torch.Tensor, reference_log_probs: torch.Tensor
) -> torch.Tensor:
    """Non-negative per-token KL estimator used by GRPO implementations."""
    log_ratio = reference_log_probs.detach() - current_log_probs
    return torch.exp(log_ratio) - log_ratio - 1.0


def clipped_grpo_loss(
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    advantage: torch.Tensor,
    clip_eps: float,
    kl_coef: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute token-level clipped GRPO loss and diagnostics."""
    if not (
        current_log_probs.shape
        == old_log_probs.shape
        == reference_log_probs.shape
    ):
        raise ValueError("Current, old and reference log-prob tensors must align.")
    if current_log_probs.numel() == 0:
        raise ValueError("Cannot optimize an empty response.")

    log_ratio = torch.clamp(
        current_log_probs - old_log_probs.detach(), min=-20.0, max=20.0
    )
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    policy_objective = torch.minimum(
        ratio * advantage, clipped_ratio * advantage
    )
    kl = sampled_reverse_kl(current_log_probs, reference_log_probs)
    loss = -policy_objective.mean() + kl_coef * kl.mean()
    diagnostics = {
        "ratio_mean": ratio.mean().detach(),
        "clip_fraction": ((ratio - 1.0).abs() > clip_eps)
        .float()
        .mean()
        .detach(),
        "approx_kl": kl.mean().detach(),
        "policy_objective": policy_objective.mean().detach(),
    }
    return loss, diagnostics
