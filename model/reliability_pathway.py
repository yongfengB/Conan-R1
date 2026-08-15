"""Tensor-level reference implementation of the Conan-R1 visual pathway.

This module implements the reliability target in Eq. (5) and the downstream
visual pathway of the revised manuscript without
depending on a particular video decoder or language backbone.  The appearance
tokens are expected to come from a frozen visual encoder.  Optical flow is
estimated outside this module and supplied at its native frame interval; the
normalization below converts displacement to velocity before applying the
training-split scale ``v_max``.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


RELIABILITY_INTERVENTIONS = {
    "predicted",
    "all_one",
    "spatial_shuffle",
    "temporal_shuffle",
    "branch_swap",
}

# Machine-readable identifiers for the exact paper target.  Checkpoint and
# training-config validation use these values so an L2 target cannot be labeled
# as the released cosine target.
RELIABILITY_TARGET_METRIC = "layernorm_cosine_angular_discrepancy"
RELIABILITY_TARGET_FORMULA = "exp(-(1-cos(LN(F_d),LN(F_r)))/(2*tau_b))"


@dataclass(frozen=True)
class ReliabilityPathwayConfig:
    appearance_dim: int
    hidden_dim: int
    output_dim: int
    degradation_dim: int
    num_factors: int
    max_anchors: int = 25
    max_spatial_tokens: int = 256
    q_min: float = 0.05
    reliability_prior_scale: float = 1.0
    target_metric: str = RELIABILITY_TARGET_METRIC
    tau_appearance: float = 0.25
    tau_motion: float = 0.25
    ema_decay: float = 0.999
    dropout: float = 0.0
    use_reliability_fusion: bool = True
    use_event_aware_pooling: bool = True
    use_temporal_reliability: bool = True

    def __post_init__(self) -> None:
        if min(
            self.appearance_dim,
            self.hidden_dim,
            self.output_dim,
            self.degradation_dim,
            self.num_factors,
        ) <= 0:
            raise ValueError("All pathway dimensions must be positive.")
        if not 0.0 < self.q_min <= 1.0:
            raise ValueError("q_min must lie in (0, 1].")
        if self.target_metric != RELIABILITY_TARGET_METRIC:
            raise ValueError(
                "The released method requires layer-normalized cosine angular "
                "discrepancy; L2 reliability targets are not compatible."
            )
        if (
            not math.isfinite(self.tau_appearance)
            or not math.isfinite(self.tau_motion)
            or self.tau_appearance <= 0.0
            or self.tau_motion <= 0.0
        ):
            raise ValueError("Teacher temperatures must be positive.")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must lie in [0, 1).")


@dataclass
class ReliabilityPathwayOutput:
    video_tokens: torch.Tensor
    local_tokens: torch.Tensor
    temporal_tokens: torch.Tensor
    appearance_reliability: torch.Tensor
    motion_reliability: torch.Tensor
    modality_gates: torch.Tensor
    event_weights: torch.Tensor
    frame_reliability: torch.Tensor
    temporal_attention: torch.Tensor
    degradation_presence_logits: torch.Tensor
    degradation_severity: torch.Tensor
    degradation_embedding: torch.Tensor


def normalize_native_motion(
    flow_displacement: torch.Tensor,
    elapsed_seconds: torch.Tensor,
    v_max: float,
) -> torch.Tensor:
    """Return ``[u_x, u_y, ||u||]`` after seconds-based normalization.

    Args:
        flow_displacement: ``[..., 2]`` pixel displacement from adjacent
            native-rate frames.
        elapsed_seconds: positive scalar or tensor broadcastable to
            ``flow_displacement[..., :1]``.
        v_max: fixed positive velocity scale estimated on the training split.
    """
    if flow_displacement.shape[-1] != 2:
        raise ValueError("flow_displacement must end in x/y components.")
    if not math.isfinite(float(v_max)) or float(v_max) <= 0.0:
        raise ValueError("v_max must be a finite positive training-split scale.")
    elapsed = torch.as_tensor(
        elapsed_seconds,
        dtype=flow_displacement.dtype,
        device=flow_displacement.device,
    )
    while elapsed.ndim < flow_displacement.ndim:
        elapsed = elapsed.unsqueeze(-1)
    if torch.any(~torch.isfinite(elapsed)) or torch.any(elapsed <= 0.0):
        raise ValueError("elapsed_seconds must be finite and strictly positive.")
    velocity = (flow_displacement / elapsed / float(v_max)).clamp(-1.0, 1.0)
    magnitude = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)
    return torch.cat((velocity, magnitude), dim=-1)


def source_relative_target(
    degraded_teacher: torch.Tensor,
    source_teacher: torch.Tensor,
    tau_b: float,
    occlusion_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Construct Eq. (5)'s detached, dimension-free retention proxy.

    This is exactly ``exp(-(1-cos(LN(F_d), LN(F_r)))/(2*tau_b))``.  A logged
    occlusion mask sets local retention to zero on the affected token grid.
    """
    if degraded_teacher.shape != source_teacher.shape:
        raise ValueError("Source and degraded teacher features must be aligned.")
    if degraded_teacher.ndim < 2:
        raise ValueError("Teacher features must include a feature dimension.")
    if not math.isfinite(float(tau_b)) or float(tau_b) <= 0.0:
        raise ValueError("tau_b must be finite and positive.")
    with torch.no_grad():
        degraded = F.layer_norm(degraded_teacher, (degraded_teacher.shape[-1],))
        source = F.layer_norm(source_teacher, (source_teacher.shape[-1],))
        similarity = F.cosine_similarity(degraded, source, dim=-1, eps=1e-8)
        angular_discrepancy = (1.0 - similarity.clamp(-1.0, 1.0)) / 2.0
        target = torch.exp(-angular_discrepancy / float(tau_b))
        if occlusion_mask is not None:
            mask = occlusion_mask.to(device=target.device, dtype=target.dtype)
            if mask.shape != target.shape:
                raise ValueError("occlusion_mask must match the token grid.")
            if torch.any((mask < 0.0) | (mask > 1.0)):
                raise ValueError("occlusion_mask values must lie in [0, 1].")
            target = target * (1.0 - mask)
        return target.detach()


def reliability_loss(
    predicted_appearance: torch.Tensor,
    predicted_motion: torch.Tensor,
    target_appearance: torch.Tensor,
    target_motion: torch.Tensor,
) -> torch.Tensor:
    """Smooth-L1 reliability loss averaged across branches and tokens."""
    for predicted, target in (
        (predicted_appearance, target_appearance),
        (predicted_motion, target_motion),
    ):
        if predicted.shape != target.shape:
            raise ValueError("Predicted and target reliability grids must align.")
    return 0.5 * (
        F.smooth_l1_loss(predicted_appearance, target_appearance)
        + F.smooth_l1_loss(predicted_motion, target_motion)
    )


def degradation_loss(
    presence_logits: torch.Tensor,
    severity_prediction: torch.Tensor,
    presence_target: torch.Tensor,
    severity_target: torch.Tensor,
) -> torch.Tensor:
    """BCE factor-presence plus active-factor Smooth-L1 severity loss."""
    if not (
        presence_logits.shape
        == severity_prediction.shape
        == presence_target.shape
        == severity_target.shape
    ):
        raise ValueError("All degradation tensors must have shape [B, K].")
    presence = presence_target.to(presence_logits.dtype)
    bce = F.binary_cross_entropy_with_logits(
        presence_logits, presence, reduction="none"
    )
    severity = F.smooth_l1_loss(
        severity_prediction, severity_target.to(severity_prediction.dtype), reduction="none"
    )
    return (bce + presence * severity).mean()


def apply_reliability_intervention(
    appearance: torch.Tensor,
    motion: torch.Tensor,
    condition: str,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Intervene only on the reliability field used by fusion and attention."""
    if appearance.shape != motion.shape or appearance.ndim != 3:
        raise ValueError("Reliability fields must share shape [B, T, P].")
    if condition not in RELIABILITY_INTERVENTIONS:
        raise ValueError(f"Unknown reliability intervention: {condition}")
    if condition == "predicted":
        return appearance, motion
    if condition == "all_one":
        return torch.ones_like(appearance), torch.ones_like(motion)
    if condition == "branch_swap":
        return motion, appearance

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    result = []
    for field in (appearance, motion):
        changed = field.clone()
        if condition == "spatial_shuffle":
            for batch_index in range(field.shape[0]):
                for time_index in range(field.shape[1]):
                    order = torch.randperm(field.shape[2], generator=generator).to(field.device)
                    changed[batch_index, time_index] = field[batch_index, time_index, order]
        else:
            for batch_index in range(field.shape[0]):
                order = torch.randperm(field.shape[1], generator=generator).to(field.device)
                changed[batch_index] = field[batch_index, order]
        result.append(changed)
    return result[0], result[1]


class EMAMotionTeacher(nn.Module):
    """No-gradient EMA copy of the trainable motion encoder."""

    def __init__(self, online_encoder: nn.Module, decay: float = 0.999) -> None:
        super().__init__()
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must lie in [0, 1).")
        self.decay = float(decay)
        self.encoder = copy.deepcopy(online_encoder).eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, online_encoder: nn.Module) -> None:
        online_state = dict(online_encoder.named_parameters())
        for name, teacher_parameter in self.encoder.named_parameters():
            teacher_parameter.mul_(self.decay).add_(
                online_state[name].detach(), alpha=1.0 - self.decay
            )
        online_buffers = dict(online_encoder.named_buffers())
        for name, teacher_buffer in self.encoder.named_buffers():
            teacher_buffer.copy_(online_buffers[name].detach())

    @torch.no_grad()
    def forward(self, motion_representation: torch.Tensor) -> torch.Tensor:
        return self.encoder(motion_representation)


class ReliabilityAwarePathway(nn.Module):
    """Reliability-regulated appearance--motion fusion and temporal adapter."""

    def __init__(self, config: ReliabilityPathwayConfig) -> None:
        super().__init__()
        self.config = config
        d = config.hidden_dim
        e = config.degradation_dim
        self.appearance_input = nn.Linear(config.appearance_dim, d)
        self.motion_encoder = nn.Sequential(
            nn.Linear(3, d), nn.GELU(), nn.LayerNorm(d), nn.Linear(d, d)
        )
        self.degradation_encoder = nn.Sequential(
            nn.Linear(2 * d, e), nn.GELU(), nn.LayerNorm(e), nn.Linear(e, e)
        )
        self.degradation_presence_head = nn.Linear(e, config.num_factors)
        self.degradation_severity_head = nn.Linear(e, config.num_factors)
        reliability_input_dim = 2 * d + e
        self.appearance_reliability_head = nn.Sequential(
            nn.Linear(reliability_input_dim, d), nn.GELU(), nn.Linear(d, 1)
        )
        self.motion_reliability_head = nn.Sequential(
            nn.Linear(reliability_input_dim, d), nn.GELU(), nn.Linear(d, 1)
        )
        self.modality_gate = nn.Sequential(
            nn.Linear(reliability_input_dim + 2, d), nn.GELU(), nn.Linear(d, 2)
        )
        self.appearance_projection = nn.Linear(d, d)
        self.motion_projection = nn.Linear(d, d)
        self.reliability_projection = nn.Linear(2, d)
        self.fusion_norm = nn.LayerNorm(d)
        self.event_score = nn.Linear(d, 1, bias=False)
        self.temporal_query = nn.Linear(d, d)
        self.temporal_key = nn.Linear(d, d)
        self.temporal_value = nn.Linear(d, d)
        self.timestamp_bias = nn.Sequential(
            nn.Linear(1, d), nn.Tanh(), nn.Linear(d, 1, bias=False)
        )
        self.spatial_embedding = nn.Parameter(
            torch.zeros(config.max_spatial_tokens + 1, d)
        )
        self.temporal_embedding = nn.Parameter(torch.zeros(config.max_anchors, d))
        self.dropout = nn.Dropout(config.dropout)
        self.projector = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, config.output_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.spatial_embedding, std=0.02)
        nn.init.normal_(self.temporal_embedding, std=0.02)

    def forward(
        self,
        appearance_tokens: torch.Tensor,
        motion_representation: torch.Tensor,
        timestamps: torch.Tensor,
        *,
        reliability_intervention: str = "predicted",
        intervention_seed: int = 42,
    ) -> ReliabilityPathwayOutput:
        """Create the ordered ``[local tokens; temporal summary]`` sequence.

        ``appearance_tokens`` has shape ``[B,T,P,D_a]`` and
        ``motion_representation`` has shape ``[B,T,P,3]``.
        """
        if appearance_tokens.ndim != 4 or motion_representation.ndim != 4:
            raise ValueError("Appearance and motion inputs must have shape [B,T,P,D].")
        if appearance_tokens.shape[:3] != motion_representation.shape[:3]:
            raise ValueError("Appearance and motion token grids must align.")
        batch, anchors, spatial, _ = appearance_tokens.shape
        if timestamps.shape != (batch, anchors):
            raise ValueError("timestamps must have shape [B, T].")
        if torch.any(timestamps[:, 1:] <= timestamps[:, :-1]):
            raise ValueError("timestamps must be strictly increasing in each video.")
        if anchors > self.config.max_anchors or spatial > self.config.max_spatial_tokens:
            raise ValueError("Input exceeds configured positional-embedding capacity.")

        appearance = self.appearance_input(appearance_tokens)
        motion = self.motion_encoder(motion_representation)
        pooled = torch.cat((appearance, motion), dim=-1).mean(dim=(1, 2))
        degradation_embedding = self.degradation_encoder(pooled)
        context = degradation_embedding[:, None, None, :].expand(
            batch, anchors, spatial, -1
        )
        joint = torch.cat((appearance, motion, context), dim=-1)
        q_appearance = torch.sigmoid(self.appearance_reliability_head(joint).squeeze(-1))
        q_motion = torch.sigmoid(self.motion_reliability_head(joint).squeeze(-1))
        q_appearance, q_motion = apply_reliability_intervention(
            q_appearance,
            q_motion,
            reliability_intervention,
            seed=intervention_seed,
        )
        if self.config.use_reliability_fusion:
            gates = torch.softmax(
                self.modality_gate(
                    torch.cat(
                        (joint, q_appearance[..., None], q_motion[..., None]),
                        dim=-1,
                    )
                ),
                dim=-1,
            )
            fused = self.fusion_norm(
                gates[..., 0, None]
                * q_appearance[..., None]
                * self.appearance_projection(appearance)
                + gates[..., 1, None]
                * q_motion[..., None]
                * self.motion_projection(motion)
                + self.reliability_projection(
                    torch.stack((q_appearance, q_motion), dim=-1)
                )
            )
        else:
            gates = torch.full(
                (*appearance.shape[:-1], 2),
                0.5,
                dtype=appearance.dtype,
                device=appearance.device,
            )
            fused = self.fusion_norm(
                0.5 * self.appearance_projection(appearance)
                + 0.5 * self.motion_projection(motion)
            )
        if self.config.use_event_aware_pooling:
            event_weights = torch.softmax(
                self.event_score(fused).squeeze(-1), dim=-1
            )
        else:
            event_weights = torch.full(
                (batch, anchors, spatial),
                1.0 / spatial,
                dtype=fused.dtype,
                device=fused.device,
            )
        content = torch.sum(event_weights[..., None] * fused, dim=2)
        frame_reliability = torch.sum(
            event_weights
            * (gates[..., 0] * q_appearance + gates[..., 1] * q_motion),
            dim=2,
        )

        query = self.temporal_query(content)
        key = self.temporal_key(content)
        value = self.temporal_value(content)
        scores = torch.einsum("btd,bud->btu", query, key) / math.sqrt(
            self.config.hidden_dim
        )
        time_delta = timestamps[:, :, None] - timestamps[:, None, :]
        scores = scores + self.timestamp_bias(time_delta[..., None]).squeeze(-1)
        if self.config.use_temporal_reliability:
            scores = scores + self.config.reliability_prior_scale * torch.log(
                frame_reliability.clamp(self.config.q_min, 1.0)
            )[:, None, :]
        temporal_attention = torch.softmax(scores, dim=-1)
        temporal = content + torch.einsum("btu,bud->btd", temporal_attention, value)

        ordered = torch.cat((fused, temporal[:, :, None, :]), dim=2)
        ordered = ordered + self.spatial_embedding[: spatial + 1][None, None, :, :]
        ordered = ordered + self.temporal_embedding[:anchors][None, :, None, :]
        video_tokens = self.projector(self.dropout(ordered)).reshape(
            batch, anchors * (spatial + 1), self.config.output_dim
        )
        return ReliabilityPathwayOutput(
            video_tokens=video_tokens,
            local_tokens=fused,
            temporal_tokens=temporal,
            appearance_reliability=q_appearance,
            motion_reliability=q_motion,
            modality_gates=gates,
            event_weights=event_weights,
            frame_reliability=frame_reliability,
            temporal_attention=temporal_attention,
            degradation_presence_logits=self.degradation_presence_head(
                degradation_embedding
            ),
            degradation_severity=torch.sigmoid(
                self.degradation_severity_head(degradation_embedding)
            ),
            degradation_embedding=degradation_embedding,
        )

    def trainable_policy_parameters(self):
        """Yield all response-changing visual-policy parameters."""
        yield from self.parameters()


class DiagnosticConsistencyReadouts(nn.Module):
    """Read TYPE/INFLUENCE decoder slots into auditable numeric variables."""

    def __init__(self, hidden_dim: int, num_factors: int) -> None:
        super().__init__()
        self.num_factors = int(num_factors)
        self.profile = nn.Linear(hidden_dim, 2 * num_factors)
        # interval start/end, appearance/motion branch weights, mean retention
        self.reliability = nn.Linear(hidden_dim, 5)

    def forward(
        self, type_slot: torch.Tensor, influence_slot: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        profile = self.profile(type_slot)
        reliability = self.reliability(influence_slot)
        return {
            "presence_logits": profile[..., : self.num_factors],
            "severity": torch.sigmoid(profile[..., self.num_factors :]),
            "interval": torch.sigmoid(reliability[..., :2]),
            "branch_weights": torch.softmax(reliability[..., 2:4], dim=-1),
            "mean_retention": torch.sigmoid(reliability[..., 4]),
        }


def consistency_loss(
    readout: Dict[str, torch.Tensor],
    internal_presence_logits: torch.Tensor,
    internal_severity: torch.Tensor,
    reliability_summary: torch.Tensor,
) -> torch.Tensor:
    """Align diagnostic text slots with internal profile/reliability variables.

    ``reliability_summary`` is ``[start, end, appearance_weight,
    motion_weight, mean_retention]`` with all entries normalized to ``[0,1]``.
    Internal control variables are detached so this loss trains the text-side
    readouts rather than changing the variables it is intended to audit.
    """
    if reliability_summary.shape[-1] != 5:
        raise ValueError("reliability_summary must end in five fields.")
    presence_target = torch.sigmoid(internal_presence_logits.detach())
    severity_target = internal_severity.detach()
    profile = F.binary_cross_entropy_with_logits(
        readout["presence_logits"], presence_target
    ) + F.smooth_l1_loss(readout["severity"], severity_target)
    predicted_summary = torch.cat(
        (
            readout["interval"],
            readout["branch_weights"],
            readout["mean_retention"][..., None],
        ),
        dim=-1,
    )
    return profile + F.smooth_l1_loss(
        predicted_summary, reliability_summary.detach()
    )


def summarize_reliability_field(
    output: ReliabilityPathwayOutput, timestamps: torch.Tensor
) -> torch.Tensor:
    """Create the five-value internal summary verbalized by INFLUENCE.

    Evidence-loss weights define a differentiable affected interval as
    ``center ± one weighted standard deviation``.  The remaining entries are
    appearance/motion contribution and mean source-relative retention.
    Timestamps are normalized by the clip duration.
    """
    if timestamps.shape != output.frame_reliability.shape:
        raise ValueError("timestamps must align with frame reliability.")
    duration = timestamps[:, -1:].clamp_min(1e-6)
    normalized_time = timestamps / duration
    evidence_loss = (1.0 - output.frame_reliability).clamp_min(1e-6)
    weights = evidence_loss / evidence_loss.sum(dim=1, keepdim=True)
    center = torch.sum(weights * normalized_time, dim=1)
    spread = torch.sqrt(
        torch.sum(weights * torch.square(normalized_time - center[:, None]), dim=1)
        + 1e-6
    )
    interval = torch.stack(
        ((center - spread).clamp(0.0, 1.0), (center + spread).clamp(0.0, 1.0)),
        dim=-1,
    )
    branch = output.modality_gates.mean(dim=(1, 2))
    retention = output.frame_reliability.mean(dim=1, keepdim=True)
    return torch.cat((interval, branch, retention), dim=-1)
