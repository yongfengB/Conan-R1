"""Public-API Qwen2.5-VL adapter for reliability-aware visual tokens.

Transformers 4.49 exposes the Qwen visual encoder and accepts ``inputs_embeds``.
This adapter freezes the visual encoder, runs the Conan-R1 pathway, and replaces
each Qwen image block by the ordered sequence ``[F_t,1:P; h_t]`` from Eq. (20).
The extra summary position is inserted before the corresponding vision-end
token, and labels, masks, input ids and multimodal RoPE indices are expanded in
lockstep.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from .reliability_pathway import (
    ReliabilityAwarePathway,
    ReliabilityPathwayOutput,
    normalize_native_motion,
)


def image_token_runs(input_ids: torch.Tensor, image_token_id: int) -> list[tuple[int, int]]:
    """Return half-open contiguous image-token runs for a batch-one prompt."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1, L].")
    positions = torch.nonzero(input_ids[0].eq(int(image_token_id)), as_tuple=False)
    positions = positions.flatten().tolist()
    if not positions:
        return []
    runs = []
    start = previous = positions[0]
    for position in positions[1:]:
        if position != previous + 1:
            runs.append((start, previous + 1))
            start = position
        previous = position
    runs.append((start, previous + 1))
    return runs


def insert_sequence_values(
    tensor: torch.Tensor,
    runs: list[tuple[int, int]],
    values: list[torch.Tensor],
) -> torch.Tensor:
    """Insert one value after every token run along dimension one."""
    if tensor.ndim < 2 or tensor.shape[0] != 1 or len(runs) != len(values):
        raise ValueError("Batch-one sequence tensor and one value per run are required.")
    pieces = []
    cursor = 0
    for (_, end), value in zip(runs, values):
        pieces.append(tensor[:, cursor:end])
        expected = (1, 1, *tensor.shape[2:])
        if tuple(value.shape) != expected:
            raise ValueError(f"Inserted value must have shape {expected}.")
        pieces.append(value.to(device=tensor.device, dtype=tensor.dtype))
        cursor = end
    pieces.append(tensor[:, cursor:])
    return torch.cat(pieces, dim=1)


@dataclass
class QwenAdaptedInputs:
    model_inputs: Dict[str, torch.Tensor]
    pathway_output: ReliabilityPathwayOutput
    appearance_tokens: torch.Tensor
    motion_representation: torch.Tensor
    input_ids: torch.Tensor
    type_decoder_slot: Optional[torch.Tensor] = None
    influence_decoder_slot: Optional[torch.Tensor] = None


class QwenReliabilityAdapter:
    """Convert processor inputs into response-changing reliability embeddings."""

    def __init__(
        self,
        policy_model: Any,
        pathway: ReliabilityAwarePathway,
        *,
        v_max: float,
    ) -> None:
        self.policy_model = policy_model
        self.pathway = pathway
        self.v_max = float(v_max)

    def _base_model(self):
        model = self.policy_model
        if hasattr(model, "get_base_model"):
            model = model.get_base_model()
        while (
            not hasattr(model, "visual")
            and hasattr(model, "model")
            and model.model is not model
        ):
            model = model.model
        if not hasattr(model, "visual"):
            raise RuntimeError("Could not resolve the Qwen2.5-VL base model.")
        return model

    def prepare(
        self,
        processor_inputs: Dict[str, torch.Tensor],
        motion_displacement_tokens: torch.Tensor,
        elapsed_seconds: torch.Tensor,
        timestamps: torch.Tensor,
        *,
        reliability_intervention: str = "predicted",
        intervention_seed: int = 42,
    ) -> QwenAdaptedInputs:
        """Build ``inputs_embeds`` with Conan-R1 visual tokens.

        The current reference supports batch size one because the public
        processor flattens all image patches across images.  This is identical
        to the paper's per-device batch size; DDP supplies the global batch.
        """
        if processor_inputs["input_ids"].shape[0] != 1:
            raise ValueError("The reference Qwen adapter requires per-device batch size 1.")
        if "pixel_values" not in processor_inputs or "image_grid_thw" not in processor_inputs:
            raise ValueError("Qwen processor inputs must contain images and image_grid_thw.")
        input_ids = processor_inputs["input_ids"]
        base = self._base_model()
        visual = base.visual
        for parameter in visual.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            image_embeds = visual(
                processor_inputs["pixel_values"].to(dtype=visual.dtype),
                grid_thw=processor_inputs["image_grid_thw"],
            )
        image_grid = processor_inputs["image_grid_thw"]
        merge = int(base.config.vision_config.spatial_merge_size)
        per_image = [
            int(t * (h // merge) * (w // merge))
            for t, h, w in image_grid.tolist()
        ]
        if len(set(per_image)) != 1:
            raise ValueError("All sampled anchors must use the same token grid.")
        anchors = len(per_image)
        spatial = per_image[0]
        appearance = image_embeds.reshape(1, anchors, spatial, -1)
        if motion_displacement_tokens.shape[:3] != (1, anchors, spatial):
            raise ValueError(
                "motion_displacement_tokens must align with Qwen's merged image grid."
            )
        motion = normalize_native_motion(
            motion_displacement_tokens, elapsed_seconds, self.v_max
        )
        pathway_parameter = next(self.pathway.parameters())
        pathway_dtype = pathway_parameter.dtype
        pathway_device = pathway_parameter.device
        appearance = appearance.to(device=pathway_device, dtype=pathway_dtype)
        motion = motion.to(device=pathway_device, dtype=pathway_dtype)
        timestamps = timestamps.to(device=pathway_device, dtype=pathway_dtype)
        pathway_output = self.pathway(
            appearance,
            motion,
            timestamps,
            reliability_intervention=reliability_intervention,
            intervention_seed=intervention_seed,
        )
        response_visual = pathway_output.video_tokens.reshape(
            anchors, spatial + 1, self.pathway.config.output_dim
        )
        text_embeddings = base.model.embed_tokens(input_ids)
        runs = image_token_runs(input_ids, base.config.image_token_id)
        if len(runs) != anchors or [end - start for start, end in runs] != per_image:
            raise ValueError("Qwen image-token runs do not match image_grid_thw.")
        local_embeddings = text_embeddings.clone()
        for anchor, (start, end) in enumerate(runs):
            local_embeddings[:, start:end] = response_visual[
                anchor, :spatial
            ][None].to(text_embeddings.dtype)
        inputs_embeds = insert_sequence_values(
            local_embeddings,
            runs,
            [
                response_visual[anchor, spatial][None, None]
                for anchor in range(anchors)
            ],
        )

        summary_token_id = int(base.config.vision_end_token_id)
        expanded_input_ids = insert_sequence_values(
            input_ids,
            runs,
            [
                torch.full((1, 1), summary_token_id, device=input_ids.device)
                for _ in runs
            ],
        )
        model_inputs = {
            key: value
            for key, value in processor_inputs.items()
            if key not in {"input_ids", "pixel_values", "attention_mask", "labels"}
        }
        if "attention_mask" in processor_inputs:
            attention_mask = insert_sequence_values(
                processor_inputs["attention_mask"],
                runs,
                [
                    torch.ones(
                        (1, 1),
                        device=processor_inputs["attention_mask"].device,
                        dtype=processor_inputs["attention_mask"].dtype,
                    )
                    for _ in runs
                ],
            )
            model_inputs["attention_mask"] = attention_mask
        if "labels" in processor_inputs:
            labels = insert_sequence_values(
                processor_inputs["labels"],
                runs,
                [
                    torch.full(
                        (1, 1),
                        -100,
                        device=processor_inputs["labels"].device,
                        dtype=processor_inputs["labels"].dtype,
                    )
                    for _ in runs
                ],
            )
            model_inputs["labels"] = labels
        model_inputs["input_ids"] = expanded_input_ids
        model_inputs["inputs_embeds"] = inputs_embeds
        position_owner = base if hasattr(base, "get_rope_index") else base.model
        position_ids, rope_deltas = position_owner.get_rope_index(
            expanded_input_ids,
            image_grid_thw=processor_inputs.get("image_grid_thw"),
            video_grid_thw=processor_inputs.get("video_grid_thw"),
            second_per_grid_ts=processor_inputs.get("second_per_grid_ts"),
            attention_mask=model_inputs.get("attention_mask"),
        )
        model_inputs["position_ids"] = position_ids
        # Supplying explicit position ids bypasses Qwen's internal assignment;
        # cache the matching delta so subsequent autoregressive steps continue
        # from the expanded prefix rather than the processor's original length.
        if hasattr(position_owner, "rope_deltas"):
            position_owner.rope_deltas = rope_deltas
        return QwenAdaptedInputs(
            model_inputs, pathway_output, appearance, motion, expanded_input_ids
        )


def pool_flow_to_token_grid(
    flow_displacement: torch.Tensor, grid_height: int, grid_width: int
) -> torch.Tensor:
    """Area-pool dense flow ``[B,T,H,W,2]`` to the Qwen merged token grid."""
    if flow_displacement.ndim != 5 or flow_displacement.shape[-1] != 2:
        raise ValueError("Dense flow must have shape [B,T,H,W,2].")
    batch, anchors, _, _, _ = flow_displacement.shape
    channels_first = flow_displacement.permute(0, 1, 4, 2, 3).reshape(
        batch * anchors, 2, flow_displacement.shape[2], flow_displacement.shape[3]
    )
    pooled = F.adaptive_avg_pool2d(channels_first, (grid_height, grid_width))
    return pooled.reshape(batch, anchors, 2, grid_height * grid_width).permute(0, 1, 3, 2)
