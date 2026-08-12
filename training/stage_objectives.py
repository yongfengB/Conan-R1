"""Stage-I and Stage-II objectives from the revised Conan-R1 manuscript."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AuxiliaryLossWeights:
    degradation: float
    reliability: float
    consistency: float

    def __post_init__(self) -> None:
        if min(self.degradation, self.reliability, self.consistency) < 0.0:
            raise ValueError("Auxiliary-loss weights must be non-negative.")


@dataclass
class StageLoss:
    total: torch.Tensor
    primary: torch.Tensor
    degradation: torch.Tensor
    reliability: torch.Tensor
    consistency: torch.Tensor


def stage1_loss(
    language_model_loss: torch.Tensor,
    degradation: torch.Tensor,
    reliability: torch.Tensor,
    consistency: torch.Tensor,
    weights: AuxiliaryLossWeights,
) -> StageLoss:
    total = (
        language_model_loss
        + weights.degradation * degradation
        + weights.reliability * reliability
        + weights.consistency * consistency
    )
    return StageLoss(
        total=total,
        primary=language_model_loss,
        degradation=degradation,
        reliability=reliability,
        consistency=consistency,
    )


def stage2_loss(
    negative_grpo_objective: torch.Tensor,
    degradation: torch.Tensor,
    reliability: torch.Tensor,
    consistency: torch.Tensor,
    weights: AuxiliaryLossWeights,
) -> StageLoss:
    total = (
        negative_grpo_objective
        + weights.degradation * degradation
        + weights.reliability * reliability
        + weights.consistency * consistency
    )
    return StageLoss(
        total=total,
        primary=negative_grpo_objective,
        degradation=degradation,
        reliability=reliability,
        consistency=consistency,
    )
