import torch
from types import SimpleNamespace

from training.stage_objectives import AuxiliaryLossWeights, stage1_loss, stage2_loss
from training.auxiliary import rasterize_logged_occlusions


def test_stage1_and_stage2_preserve_auxiliary_losses():
    weights = AuxiliaryLossWeights(1.0, 0.5, 0.25)
    pieces = [torch.tensor(value) for value in (2.0, 1.0, 4.0, 8.0)]
    sft = stage1_loss(*pieces, weights)
    rl = stage2_loss(*pieces, weights)
    assert sft.total.item() == 7.0
    assert rl.total.item() == 7.0


def test_occlusion_rasterization_uses_source_anchor_indices():
    output = SimpleNamespace(
        appearance_reliability=torch.zeros(1, 2, 4)
    )
    boxes = [None] * 11
    boxes[10] = [0.5, 0.5, 1.0, 1.0]
    mask = rasterize_logged_occlusions(
        {
            "anchor_indices": [0, 10],
            "synthesis_metadata": {
                "occlusion_boxes_norm_by_frame": {"vehicle_mask": boxes}
            },
        },
        output,
        "cpu",
    )
    assert mask[0, 0].sum() == 0
    assert mask[0, 1].view(2, 2)[1, 1] == 1
