"""Source-paired auxiliary losses shared by SFT and GRPO."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch

from model.reliability_pathway import (
    DiagnosticConsistencyReadouts,
    EMAMotionTeacher,
    ReliabilityPathwayOutput,
    consistency_loss,
    degradation_loss,
    reliability_loss,
    source_relative_target,
    summarize_reliability_field,
)


@dataclass
class AuxiliaryBatch:
    factor_presence: torch.Tensor
    factor_severity: torch.Tensor
    source_appearance_teacher: torch.Tensor
    degraded_appearance_teacher: torch.Tensor
    source_motion_representation: torch.Tensor
    degraded_motion_representation: torch.Tensor
    occlusion_token_mask: torch.Tensor
    timestamps: torch.Tensor
    type_decoder_slot: torch.Tensor
    influence_decoder_slot: torch.Tensor


@dataclass
class AuxiliaryLosses:
    degradation: torch.Tensor
    reliability: torch.Tensor
    consistency: torch.Tensor


def encode_degradation_profile(
    profiles: Sequence[Sequence[Tuple[str, float]]],
    factor_names: Sequence[str],
    *,
    device: torch.device | str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode logged factor presence and severity with a frozen vocabulary."""
    lookup = {name: index for index, name in enumerate(factor_names)}
    presence = torch.zeros(len(profiles), len(factor_names), device=device)
    severity = torch.zeros_like(presence)
    for batch_index, profile in enumerate(profiles):
        for factor, value in profile:
            if factor not in lookup:
                raise ValueError(f"Unknown degradation factor: {factor}")
            factor_index = lookup[factor]
            presence[batch_index, factor_index] = 1.0
            severity[batch_index, factor_index] = float(value)
    return presence, severity


def rasterize_logged_occlusions(
    sample: dict,
    output: ReliabilityPathwayOutput,
    device: torch.device | str,
) -> torch.Tensor:
    """Map source-frame occlusion logs onto the exact sampled anchor grid."""
    batch, anchors, spatial = output.appearance_reliability.shape
    if batch != 1:
        raise ValueError("Reference occlusion rasterization requires batch size one.")
    side = int(round(spatial ** 0.5))
    if side * side != spatial:
        raise ValueError("Reference occlusion rasterization requires a square grid.")
    anchor_indices = [int(value) for value in sample.get("anchor_indices", [])]
    if len(anchor_indices) != anchors:
        raise ValueError("anchor_indices must align with the reliability grid.")
    mask = torch.zeros(batch, anchors, spatial, device=device)
    logs = sample.get("synthesis_metadata", {}).get(
        "occlusion_boxes_norm_by_frame", {}
    )
    for boxes in logs.values():
        for anchor_slot, source_frame_index in enumerate(anchor_indices):
            if source_frame_index >= len(boxes):
                raise ValueError("Occlusion log is shorter than a sampled anchor index.")
            box = boxes[source_frame_index]
            if box is None:
                continue
            x1, y1, x2, y2 = map(float, box)
            if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
                raise ValueError("Logged occlusion boxes must be normalized and valid.")
            left = max(0, min(side, int(x1 * side)))
            right = max(left + 1, min(side, int(math.ceil(x2 * side))))
            top = max(0, min(side, int(y1 * side)))
            bottom = max(top + 1, min(side, int(math.ceil(y2 * side))))
            mask[0, anchor_slot].view(side, side)[top:bottom, left:right] = 1.0
    return mask


def compute_auxiliary_losses(
    output: ReliabilityPathwayOutput,
    batch: AuxiliaryBatch,
    *,
    motion_teacher: EMAMotionTeacher,
    consistency_readouts: DiagnosticConsistencyReadouts,
    tau_appearance: float,
    tau_motion: float,
) -> AuxiliaryLosses:
    """Compute L_deg, L_rel and L_cons for the current policy forward."""
    with torch.no_grad():
        source_motion = motion_teacher(batch.source_motion_representation)
        degraded_motion = motion_teacher(batch.degraded_motion_representation)
    target_appearance = source_relative_target(
        batch.degraded_appearance_teacher,
        batch.source_appearance_teacher,
        tau_appearance,
        batch.occlusion_token_mask,
    )
    target_motion = source_relative_target(
        degraded_motion,
        source_motion,
        tau_motion,
        batch.occlusion_token_mask,
    )
    rel = reliability_loss(
        output.appearance_reliability,
        output.motion_reliability,
        target_appearance,
        target_motion,
    )
    deg = degradation_loss(
        output.degradation_presence_logits,
        output.degradation_severity,
        batch.factor_presence,
        batch.factor_severity,
    )
    readout = consistency_readouts(
        batch.type_decoder_slot, batch.influence_decoder_slot
    )
    summary = summarize_reliability_field(output, batch.timestamps)
    cons = consistency_loss(
        readout,
        output.degradation_presence_logits,
        output.degradation_severity,
        summary,
    )
    return AuxiliaryLosses(deg, rel, cons)
