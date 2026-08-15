"""Equation-level tests for the revised reliability-aware pathway."""
import math

import pytest
import torch

from model.reliability_pathway import (
    EMAMotionTeacher,
    ReliabilityAwarePathway,
    ReliabilityPathwayConfig,
    apply_reliability_intervention,
    degradation_loss,
    normalize_native_motion,
    reliability_loss,
    source_relative_target,
)


def _config():
    return ReliabilityPathwayConfig(
        appearance_dim=8,
        hidden_dim=12,
        output_dim=16,
        degradation_dim=6,
        num_factors=4,
        max_anchors=5,
        max_spatial_tokens=9,
        dropout=0.0,
    )


def test_native_motion_uses_elapsed_seconds_and_fixed_scale():
    displacement = torch.tensor([[[[2.0, -1.0]]]])
    representation = normalize_native_motion(
        displacement, torch.tensor([[[0.5]]]), v_max=4.0
    )
    assert representation.shape[-1] == 3
    assert torch.allclose(representation[..., :2], torch.tensor([[[[1.0, -0.5]]]]))
    assert torch.allclose(
        representation[..., 2], torch.sqrt(torch.tensor(1.25)).reshape(1, 1, 1)
    )


def test_source_relative_target_is_detached_and_masks_occlusion():
    source = torch.randn(1, 2, 3, 8, requires_grad=True)
    degraded = source.detach().clone().requires_grad_(True)
    mask = torch.zeros(1, 2, 3)
    mask[:, 0, 1] = 1.0
    target = source_relative_target(degraded, source, 0.25, mask)
    assert target.requires_grad is False
    assert target[0, 0, 1] == 0.0
    assert target[0, 1, 1] == 1.0


def test_source_relative_target_is_dimension_invariant():
    source = torch.tensor([[[[1.0, -1.0, 1.0, -1.0]]]])
    degraded = torch.tensor([[[[1.0, -1.0, -1.0, 1.0]]]])
    small = source_relative_target(degraded, source, 0.25)
    large = source_relative_target(
        degraded.repeat(1, 1, 1, 8), source.repeat(1, 1, 1, 8), 0.25
    )
    assert torch.allclose(small, large, atol=1e-6)


def test_source_relative_target_matches_paper_eq5_exactly():
    source = torch.tensor([[[[1.0, -1.0]]]])
    degraded = -source
    target = source_relative_target(degraded, source, tau_b=0.25)
    # cos(LN(F_d), LN(F_r)) = -1, so Eq. (5) gives exp(-1 / 0.25).
    assert target.item() == pytest.approx(math.exp(-4.0), rel=1e-6)


@pytest.mark.parametrize("tau_b", [0.0, -0.25, float("nan"), float("inf")])
def test_source_relative_target_rejects_invalid_tau(tau_b):
    features = torch.tensor([[[[1.0, -1.0]]]])
    with pytest.raises(ValueError):
        source_relative_target(features, features, tau_b=tau_b)


def test_reliability_pathway_shapes_and_normalizations():
    pathway = ReliabilityAwarePathway(_config())
    appearance = torch.randn(2, 5, 9, 8)
    motion = torch.randn(2, 5, 9, 3)
    timestamps = torch.arange(5).float().repeat(2, 1) / 4.0
    output = pathway(appearance, motion, timestamps)
    assert output.video_tokens.shape == (2, 50, 16)
    assert output.local_tokens.shape == (2, 5, 9, 12)
    assert output.temporal_tokens.shape == (2, 5, 12)
    assert torch.allclose(output.modality_gates.sum(-1), torch.ones(2, 5, 9))
    assert torch.allclose(output.event_weights.sum(-1), torch.ones(2, 5))
    assert torch.allclose(output.temporal_attention.sum(-1), torch.ones(2, 5))
    assert torch.all((output.frame_reliability >= 0) & (output.frame_reliability <= 1))


def test_cumulative_architecture_switches_change_the_executed_path():
    config = ReliabilityPathwayConfig(
        appearance_dim=8,
        hidden_dim=12,
        output_dim=16,
        degradation_dim=6,
        num_factors=2,
        max_anchors=3,
        max_spatial_tokens=4,
        use_reliability_fusion=False,
        use_event_aware_pooling=False,
        use_temporal_reliability=False,
    )
    output = ReliabilityAwarePathway(config)(
        torch.randn(1, 3, 4, 8),
        torch.randn(1, 3, 4, 3),
        torch.tensor([[0.0, 0.5, 1.0]]),
    )
    assert torch.allclose(
        output.modality_gates, torch.full_like(output.modality_gates, 0.5)
    )
    assert torch.allclose(
        output.event_weights, torch.full_like(output.event_weights, 0.25)
    )


def test_interventions_preserve_shape_and_shuffle_values():
    qa = torch.arange(24).reshape(1, 3, 8).float() / 24
    qm = 1.0 - qa
    for condition in (
        "predicted",
        "all_one",
        "spatial_shuffle",
        "temporal_shuffle",
        "branch_swap",
    ):
        changed_a, changed_m = apply_reliability_intervention(
            qa, qm, condition, seed=17
        )
        assert changed_a.shape == qa.shape
        assert changed_m.shape == qm.shape
    spatial, _ = apply_reliability_intervention(qa, qm, "spatial_shuffle", seed=17)
    assert torch.equal(torch.sort(spatial.flatten()).values, torch.sort(qa.flatten()).values)
    swapped_a, swapped_m = apply_reliability_intervention(qa, qm, "branch_swap")
    assert torch.equal(swapped_a, qm)
    assert torch.equal(swapped_m, qa)


def test_auxiliary_losses_are_finite_and_active_factor_masked():
    presence_logits = torch.zeros(1, 3, requires_grad=True)
    severity = torch.tensor([[0.9, 0.8, 0.7]], requires_grad=True)
    presence_target = torch.tensor([[1.0, 0.0, 0.0]])
    severity_target = torch.tensor([[0.2, 0.0, 0.0]])
    loss = degradation_loss(
        presence_logits, severity, presence_target, severity_target
    )
    loss.backward()
    assert severity.grad[0, 0].abs() > 0
    assert severity.grad[0, 1] == 0
    assert severity.grad[0, 2] == 0
    rel = reliability_loss(
        torch.ones(1, 2, 3),
        torch.ones(1, 2, 3),
        torch.zeros(1, 2, 3),
        torch.zeros(1, 2, 3),
    )
    assert torch.isfinite(rel)


def test_ema_teacher_updates_without_gradients():
    online = torch.nn.Linear(3, 4)
    teacher = EMAMotionTeacher(online, decay=0.5)
    before = teacher.encoder.weight.clone()
    with torch.no_grad():
        online.weight.add_(2.0)
    teacher.update(online)
    assert torch.allclose(teacher.encoder.weight, before + 1.0)
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
