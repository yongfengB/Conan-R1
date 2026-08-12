# training package
from .stage_objectives import (
    AuxiliaryLossWeights,
    StageLoss,
    stage1_loss,
    stage2_loss,
)

__all__ = [
    "AuxiliaryLossWeights",
    "StageLoss",
    "stage1_loss",
    "stage2_loss",
]
