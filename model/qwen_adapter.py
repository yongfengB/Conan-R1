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


DIAGNOSTIC_MARKERS = ("<TYPE>", "<INFLUENCE>")
DIAGNOSTIC_SLOT_PROTOCOL = "label_mask_response_fast_offsets_v1"


class DiagnosticSlotError(RuntimeError):
    """Raised when a diagnostic slot cannot be resolved inside the response."""


def response_only_labels(
    tokenizer: Any,
    response: str,
    eos_text: str,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mask the prompt by aligning the complete response suffix exactly."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise DiagnosticSlotError("Response label construction requires [1, L] ids.")
    expected = tokenizer(
        response + eos_text, add_special_tokens=False
    )["input_ids"]
    if expected and isinstance(expected[0], list):
        expected = expected[0]
    if not expected:
        raise DiagnosticSlotError("The tokenizer produced no response target ids.")
    if attention_mask is None:
        active_positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    else:
        if attention_mask.shape != input_ids.shape:
            raise DiagnosticSlotError("attention_mask must align with input_ids.")
        active_positions = torch.nonzero(
            attention_mask[0].ne(0), as_tuple=False
        ).flatten()
    if len(expected) > active_positions.numel():
        raise DiagnosticSlotError("The response target is longer than the active sequence.")
    response_positions = active_positions[-len(expected) :]
    actual = input_ids[0, response_positions].tolist()
    if actual != list(expected):
        raise DiagnosticSlotError(
            "Full processor ids do not end with the exact response target."
        )
    labels = torch.full_like(input_ids, -100)
    labels[0, response_positions] = input_ids[0, response_positions]
    return labels


def response_marker_hidden_positions(
    tokenizer: Any,
    response: str,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    markers: tuple[str, ...] = DIAGNOSTIC_MARKERS,
) -> dict[str, int]:
    """Map response marker ends to decoder positions using fast-tokenizer offsets.

    ``labels`` is the authoritative prompt/response boundary: only positions whose
    labels are not ``-100`` can own a diagnostic slot.  The complete response is
    tokenized once, so context-sensitive BPE merges between adjacent structured
    blocks are preserved; isolated marker token ids are never searched.
    """
    if input_ids.ndim != 2 or labels.ndim != 2 or input_ids.shape != labels.shape:
        raise DiagnosticSlotError(
            "input_ids and labels must have the same batch-one [1, L] shape."
        )
    if input_ids.shape[0] != 1:
        raise DiagnosticSlotError("Diagnostic slot extraction requires batch size one.")
    response_positions = torch.nonzero(labels[0].ne(-100), as_tuple=False).flatten()
    if response_positions.numel() == 0:
        raise DiagnosticSlotError("The label mask contains no supervised response tokens.")
    encoded = tokenizer(
        response,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    if "offset_mapping" not in encoded:
        raise DiagnosticSlotError(
            "A fast tokenizer with return_offsets_mapping support is required."
        )
    response_ids = list(encoded["input_ids"])
    offsets = [tuple(int(value) for value in pair) for pair in encoded["offset_mapping"]]
    if response_ids and isinstance(response_ids[0], list):
        if len(response_ids) != 1:
            raise DiagnosticSlotError("Response tokenization must produce one sequence.")
        response_ids = response_ids[0]
        offsets = offsets[0]
    if not response_ids or len(response_ids) != len(offsets):
        raise DiagnosticSlotError("Response ids and character offsets are empty or misaligned.")
    if len(response_ids) > response_positions.numel():
        raise DiagnosticSlotError(
            "The label mask is shorter than the independently tokenized response."
        )
    labelled_ids = input_ids[0, response_positions[: len(response_ids)]].tolist()
    if labelled_ids != response_ids:
        raise DiagnosticSlotError(
            "The label mask does not begin at the exact response-token boundary."
        )

    positions: dict[str, int] = {}
    for marker in markers:
        starts = [
            index
            for index in range(len(response))
            if response.startswith(marker, index)
        ]
        if len(starts) != 1:
            raise DiagnosticSlotError(
                f"Expected one {marker} marker in the supervised response, "
                f"found {len(starts)}."
            )
        marker_end = starts[0] + len(marker)
        relative_slot = next(
            (
                index
                for index, (start, end) in enumerate(offsets)
                if end > start
                and (start >= marker_end or start <= marker_end < end)
            ),
            None,
        )
        if relative_slot is None:
            raise DiagnosticSlotError(
                f"No decoder token follows the {marker} marker in the response."
            )
        positions[marker] = int(response_positions[relative_slot].item())
    return positions


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
