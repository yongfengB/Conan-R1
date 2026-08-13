"""Qwen2.5-VL-3B-Instruct policy used by Conan-R1."""
from __future__ import annotations
from contextlib import nullcontext
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from dataset.video_utils import (
    FARNEBACK_PARAMETERS,
    farneback_pair_flow,
    validate_farneback_parameters,
)
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .parser import StructuredOutput, parse_structured_output
from .reliability_pathway import (
    DiagnosticConsistencyReadouts,
    EMAMotionTeacher,
    ReliabilityAwarePathway,
    ReliabilityPathwayConfig,
)
from .qwen_adapter import QwenReliabilityAdapter, pool_flow_to_token_grid

logger = logging.getLogger(__name__)
QWEN_BASE_REVISION = "c747f21f03e7d0792c30766310bd7d8de17eeeb3"


# ---------------------------------------------------------------------------
# LoRA configuration
# ---------------------------------------------------------------------------

@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Optional[List[str]] = None

    def to_peft_config(self) -> LoraConfig:
        kwargs = dict(
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        kwargs["target_modules"] = self.target_modules or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        return LoraConfig(**kwargs)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ConanR1Model:
    """Qwen2.5-VL-3B-Instruct policy for structured five-block generation."""

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        base_model_revision: Optional[str] = QWEN_BASE_REVISION,
        lora_config: Optional[LoRAConfig] = None,
        device: Optional[str] = None,
        enable_lora: bool = True,
        reliability_config: Optional[ReliabilityPathwayConfig] = None,
        motion_v_max: Optional[float] = None,
        degradation_factor_names: Optional[List[str]] = None,
        motion_flow_parameters: Optional[dict] = None,
    ) -> None:
        self.base_model_name = base_model
        self.base_model_revision = base_model_revision
        self.lora_config = lora_config or LoRAConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.startswith("cuda") else torch.float32

        logger.info("Loading base model: %s", base_model)
        self.processor = AutoProcessor.from_pretrained(
            base_model,
            revision=base_model_revision,
            trust_remote_code=True,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model,
            revision=base_model_revision,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )
        self.enable_lora = enable_lora
        if enable_lora:
            self.model = get_peft_model(
                self.model, self.lora_config.to_peft_config()
            )
        self.model.to(self.device)
        self.reliability_pathway = None
        self.motion_teacher = None
        self.consistency_readouts = None
        self.reliability_adapter = None
        self.motion_v_max = motion_v_max
        self.motion_flow_parameters = validate_farneback_parameters(
            motion_flow_parameters or FARNEBACK_PARAMETERS
        )
        self.degradation_factor_names = list(degradation_factor_names or [])
        if reliability_config is not None:
            if len(self.degradation_factor_names) != reliability_config.num_factors:
                raise ValueError(
                    "degradation_factor_names must match reliability_config.num_factors."
                )
            self.attach_reliability_pathway(reliability_config)
        if enable_lora:
            logger.info(
                "Model loaded on %s with LoRA (rank=%d, alpha=%d).",
                self.device,
                self.lora_config.rank,
                self.lora_config.alpha,
            )
        else:
            logger.info("Unadapted base model loaded on %s.", self.device)

    def _policy_model(self):
        """Return the PEFT model, unwrapping DDP when needed."""
        if isinstance(self.model, DistributedDataParallel):
            return self.model.module
        return self.model

    def enable_distributed(self, local_rank: int) -> None:
        """Wrap the trainable policy in DistributedDataParallel."""
        if isinstance(self.model, DistributedDataParallel):
            return
        self.model = DistributedDataParallel(
            self.model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    def synchronize_visual_gradients(self) -> None:
        """Average non-DDP visual/readout gradients across all ranks."""
        if not (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            return
        world_size = torch.distributed.get_world_size()
        for module in (self.reliability_pathway, self.consistency_readouts):
            if module is None:
                continue
            for parameter in module.parameters():
                if parameter.grad is None:
                    continue
                torch.distributed.all_reduce(
                    parameter.grad, op=torch.distributed.ReduceOp.SUM
                )
                parameter.grad.div_(world_size)

    def enable_gradient_checkpointing(self) -> None:
        """Reduce activation memory for the 25-frame training protocol."""
        policy_model = self._policy_model()
        policy_model.gradient_checkpointing_enable()
        if hasattr(policy_model, "enable_input_require_grads"):
            policy_model.enable_input_require_grads()
        policy_model.config.use_cache = False

    def disable_dropout(self) -> None:
        """Disable stochastic dropout during on-policy probability evaluation.

        PPO/GRPO ratios must compare the same sampled actions under deterministic
        old and current probability evaluations.  Sampling remains stochastic
        through temperature/top-p decoding.
        """
        for module in self._policy_model().modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        for module in (self.reliability_pathway, self.consistency_readouts):
            if module is None:
                continue
            for child in module.modules():
                if isinstance(child, torch.nn.Dropout):
                    child.p = 0.0

    def attach_reliability_pathway(
        self, config: ReliabilityPathwayConfig
    ) -> None:
        """Attach all revised-method visual modules to the response policy.

        ``video_tokens`` returned by the pathway must be inserted into the
        Qwen visual-token positions by the experiment-specific adapter.  This
        explicit interface keeps the audit implementation independent of
        private Qwen forks while ensuring that every listed visual module is
        part of the response-changing policy in a full run.
        """
        self.reliability_pathway = ReliabilityAwarePathway(config).to(
            device=self.device, dtype=self.dtype
        )
        self.motion_teacher = EMAMotionTeacher(
            self.reliability_pathway.motion_encoder, decay=config.ema_decay
        ).to(device=self.device, dtype=self.dtype)
        self.consistency_readouts = DiagnosticConsistencyReadouts(
            config.output_dim, config.num_factors
        ).to(device=self.device, dtype=self.dtype)
        if self.motion_v_max is None or float(self.motion_v_max) <= 0.0:
            raise ValueError(
                "A fixed positive training-split motion_v_max is required."
            )
        self.reliability_adapter = QwenReliabilityAdapter(
            self._policy_model(), self.reliability_pathway, v_max=self.motion_v_max
        )

    def auxiliary_control_losses(
        self,
        degraded,
        source,
        *,
        factor_presence: torch.Tensor,
        factor_severity: torch.Tensor,
        occlusion_token_mask: torch.Tensor,
        timestamps: torch.Tensor,
        compute_consistency: bool = True,
        motion_target_mode: str = "ema",
        occlusion_mask_adjustment: bool = True,
    ):
        """Compute the three visual constraints from paired pathway forwards."""
        from .reliability_pathway import (
            consistency_loss,
            degradation_loss,
            reliability_loss,
            source_relative_target,
            summarize_reliability_field,
        )

        if self.motion_teacher is None or self.consistency_readouts is None:
            raise RuntimeError("Auxiliary losses require the complete reliability policy.")
        config = self.reliability_pathway.config
        if motion_target_mode not in {"ema", "online", "frozen_initial"}:
            raise ValueError(
                "motion_target_mode must be ema, online, or frozen_initial."
            )
        target_mask = occlusion_token_mask if occlusion_mask_adjustment else None
        appearance_target = source_relative_target(
            degraded.appearance_tokens,
            source.appearance_tokens,
            config.tau_appearance,
            target_mask,
        )
        with torch.no_grad():
            motion_encoder = (
                self.reliability_pathway.motion_encoder
                if motion_target_mode == "online"
                else self.motion_teacher
            )
            degraded_motion_teacher = motion_encoder(degraded.motion_representation)
            source_motion_teacher = motion_encoder(source.motion_representation)
        motion_target = source_relative_target(
            degraded_motion_teacher,
            source_motion_teacher,
            config.tau_motion,
            target_mask,
        )
        rel = reliability_loss(
            degraded.pathway_output.appearance_reliability,
            degraded.pathway_output.motion_reliability,
            appearance_target,
            motion_target,
        )
        deg = degradation_loss(
            degraded.pathway_output.degradation_presence_logits,
            degraded.pathway_output.degradation_severity,
            factor_presence,
            factor_severity,
        )
        if not compute_consistency:
            return deg, rel, deg.new_zeros(())
        if degraded.type_decoder_slot is None or degraded.influence_decoder_slot is None:
            raise RuntimeError(
                "Consistency loss requires decoder states at TYPE and INFLUENCE slots."
            )
        readout = self.consistency_readouts(
            degraded.type_decoder_slot, degraded.influence_decoder_slot
        )
        summary = summarize_reliability_field(
            degraded.pathway_output, timestamps
        )
        cons = consistency_loss(
            readout,
            degraded.pathway_output.degradation_presence_logits,
            degraded.pathway_output.degradation_severity,
            summary,
        )
        return deg, rel, cons

    def _dense_flow(
        self, frames: List[np.ndarray], motion_frames: List[np.ndarray]
    ) -> torch.Tensor:
        if len(frames) != len(motion_frames) or not frames:
            raise ValueError("Each anchor needs one adjacent native-rate motion frame.")
        flows = []
        for first, second in zip(frames, motion_frames):
            flows.append(
                farneback_pair_flow(first, second, self.motion_flow_parameters)
            )
        return torch.from_numpy(np.stack(flows).astype(np.float32))[None]

    def _adapt_reliability_inputs(
        self,
        inputs,
        frames: List[np.ndarray],
        *,
        motion_frames: Optional[List[np.ndarray]],
        elapsed_seconds: Optional[List[float]],
        timestamps: Optional[List[float]],
        reliability_intervention: str = "predicted",
        intervention_seed: int = 42,
    ):
        if self.reliability_adapter is None:
            return inputs, None
        if motion_frames is None or elapsed_seconds is None or timestamps is None:
            raise ValueError(
                "Reliability-aware inference requires native adjacent frames, "
                "elapsed seconds, and anchor timestamps."
            )
        if not (len(frames) == len(motion_frames) == len(elapsed_seconds) == len(timestamps)):
            raise ValueError("Visual context lists must have one entry per anchor.")
        image_grid = inputs["image_grid_thw"]
        merge = int(
            self.reliability_adapter._base_model().config.vision_config.spatial_merge_size
        )
        first_grid = image_grid[0].tolist()
        grid_height = int(first_grid[0] * (first_grid[1] // merge))
        grid_width = int(first_grid[2] // merge)
        dense_flow = self._dense_flow(frames, motion_frames).to(self.device)
        token_flow = pool_flow_to_token_grid(dense_flow, grid_height, grid_width)
        adapted = self.reliability_adapter.prepare(
            inputs,
            token_flow,
            torch.tensor([elapsed_seconds], device=self.device, dtype=token_flow.dtype),
            torch.tensor([timestamps], device=self.device, dtype=token_flow.dtype),
            reliability_intervention=reliability_intervention,
            intervention_seed=intervention_seed,
        )
        return adapted.model_inputs, adapted

    def trainable_policy_named_parameters(
        self, scope: str = "full"
    ) -> Iterable[tuple[str, torch.nn.Parameter]]:
        """Yield LoRA-only or full response-changing policy parameters."""
        if scope not in {"full", "lora_only"}:
            raise ValueError("scope must be 'full' or 'lora_only'.")
        for name, parameter in self.model.named_parameters():
            if "lora_" in name and parameter.requires_grad:
                yield f"lora.{name}", parameter
        if scope == "lora_only":
            return
        for prefix, module in (
            ("visual", self.reliability_pathway),
            ("readout", self.consistency_readouts),
        ):
            if module is None:
                raise RuntimeError(
                    "Full-policy optimization requires an attached reliability pathway."
                )
            for name, parameter in module.named_parameters():
                if parameter.requires_grad:
                    yield f"{prefix}.{name}", parameter

    @torch.no_grad()
    def update_motion_teacher(self) -> None:
        if self.motion_teacher is None or self.reliability_pathway is None:
            raise RuntimeError("No EMA motion teacher is attached.")
        self.motion_teacher.update(self.reliability_pathway.motion_encoder)

    def save_core(self, checkpoint_path: str) -> None:
        """Save LoRA plus every revised-method trainable/EMA module."""
        self.save_lora(checkpoint_path)
        if self.reliability_pathway is None or self.consistency_readouts is None:
            raise RuntimeError("Cannot save a full core checkpoint without visual modules.")
        metadata = {
            "method": "conan-r1-source-relative-reliability-v2",
            "format_version": 4,
            "base_model": self.base_model_name,
            "base_model_revision": self.base_model_revision,
            "reliability_config": asdict(self.reliability_pathway.config),
            "motion_v_max": float(self.motion_v_max),
            "motion_flow_parameters": self.motion_flow_parameters,
            "degradation_factor_names": list(self.degradation_factor_names),
        }
        (Path(checkpoint_path) / "conan_core_config.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        payload = {
            "reliability_pathway": self.reliability_pathway.state_dict(),
            "consistency_readouts": self.consistency_readouts.state_dict(),
            "motion_teacher": (
                self.motion_teacher.state_dict() if self.motion_teacher is not None else None
            ),
        }
        torch.save(payload, Path(checkpoint_path) / "conan_core.pt")

    def load_core(self, checkpoint_path: str, is_trainable: bool = True) -> None:
        """Load a complete Stage-I/Stage-II policy checkpoint."""
        core_path = Path(checkpoint_path) / "conan_core.pt"
        metadata_path = Path(checkpoint_path) / "conan_core_config.json"
        if not core_path.is_file():
            raise FileNotFoundError(f"Complete core state not found: {core_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Core protocol metadata not found: {metadata_path}")
        if self.reliability_pathway is None or self.consistency_readouts is None:
            raise RuntimeError("Attach the configured reliability pathway before loading.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format_version") != 4:
            raise ValueError(
                "Core checkpoint lacks the v4 flow-bound protocol metadata; it cannot be "
                "safely matched to the revised method."
            )
        expected = {
            "method": "conan-r1-source-relative-reliability-v2",
            "base_model": self.base_model_name,
            "base_model_revision": self.base_model_revision,
            "reliability_config": asdict(self.reliability_pathway.config),
            "motion_v_max": float(self.motion_v_max),
            "motion_flow_parameters": self.motion_flow_parameters,
            "degradation_factor_names": list(self.degradation_factor_names),
        }
        mismatches = {
            key: {"checkpoint": metadata.get(key), "runtime": value}
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "Checkpoint protocol does not match runtime configuration: "
                f"{mismatches}"
            )
        self.load_lora(checkpoint_path, is_trainable=is_trainable)
        payload = torch.load(
            core_path, map_location=self.device, weights_only=True
        )
        self.reliability_pathway.load_state_dict(payload["reliability_pathway"])
        self.consistency_readouts.load_state_dict(payload["consistency_readouts"])
        if self.motion_teacher is not None and payload.get("motion_teacher") is not None:
            self.motion_teacher.load_state_dict(payload["motion_teacher"])
        if not is_trainable:
            for module in (self.reliability_pathway, self.consistency_readouts):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

    # ------------------------------------------------------------------
    # Frame preprocessing
    # ------------------------------------------------------------------

    def _frames_to_pil(self, frames: List[np.ndarray]) -> List[Image.Image]:
        return [Image.fromarray(f.astype(np.uint8)) for f in frames]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        frames: List[np.ndarray],
        prompt: str,
        max_new_tokens: int = 384,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        motion_frames: Optional[List[np.ndarray]] = None,
        elapsed_seconds: Optional[List[float]] = None,
        timestamps: Optional[List[float]] = None,
        reliability_intervention: str = "predicted",
        intervention_seed: int = 42,
    ) -> str:
        """Generate a structured output string for the given frames and prompt.

        Args:
            frames: List of RGB numpy arrays (H x W x C).
            prompt: Task prompt string.
            max_new_tokens: Maximum number of tokens to generate (default 384).

        Returns:
            Raw generated string (may or may not be parseable).
        """
        pil_images = self._frames_to_pil(frames)

        # Build the conversation format expected by Qwen2.5-VL.
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in pil_images],
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=pil_images or None,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        inputs, _ = self._adapt_reliability_inputs(
            inputs,
            frames,
            motion_frames=motion_frames,
            elapsed_seconds=elapsed_seconds,
            timestamps=timestamps,
            reliability_intervention=reliability_intervention,
            intervention_seed=intervention_seed,
        )

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "use_cache": True,
        }
        if do_sample:
            generation_kwargs.update(temperature=temperature, top_p=top_p)

        policy_model = self._policy_model()
        was_training = policy_model.training
        policy_model.eval()
        with torch.inference_mode():
            output_ids = policy_model.generate(
                **inputs,
                **generation_kwargs,
            )
        if was_training:
            policy_model.train()

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        # Transformers generation may return only generated ids for a prefix
        # supplied through inputs_embeds.  Slice the prefix only when it is
        # actually present in the returned sequence.
        returned_prompt = (
            output_ids.shape[1] >= input_len
            and torch.equal(output_ids[:, :input_len], inputs["input_ids"])
        )
        generated = output_ids[:, input_len:] if returned_prompt else output_ids
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0]

    def generate_structured(
        self,
        frames: List[np.ndarray],
        prompt: str,
        max_new_tokens: int = 384,
        **visual_context,
    ) -> Optional[StructuredOutput]:
        """Generate and parse a structured output.

        Returns None if the output cannot be parsed.
        """
        raw = self.generate(
            frames, prompt, max_new_tokens=max_new_tokens, **visual_context
        )
        return parse_structured_output(raw)

    # ------------------------------------------------------------------
    # LoRA checkpoint management
    # ------------------------------------------------------------------

    def save_lora(self, checkpoint_path: str) -> None:
        """Save only the LoRA adapter weights."""
        if not self.enable_lora:
            raise RuntimeError("Cannot save a LoRA adapter when LoRA is disabled.")
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
        self._policy_model().save_pretrained(checkpoint_path)
        self.processor.save_pretrained(checkpoint_path)
        logger.info("LoRA checkpoint saved to %s", checkpoint_path)

    def load_lora(self, checkpoint_path: str, is_trainable: bool = True) -> None:
        """Replace the initialized adapter with weights from ``checkpoint_path``."""
        if not self.enable_lora:
            raise RuntimeError(
                "Instantiate ConanR1Model with enable_lora=True before loading an adapter."
            )
        if isinstance(self.model, DistributedDataParallel):
            raise RuntimeError("Load the LoRA checkpoint before enabling DDP.")
        base_model = self.model.unload()
        self.model = PeftModel.from_pretrained(
            base_model,
            checkpoint_path,
            is_trainable=is_trainable,
        )
        self.model.to(self.device)
        if self.reliability_pathway is not None:
            self.reliability_adapter = QwenReliabilityAdapter(
                self._policy_model(),
                self.reliability_pathway,
                v_max=float(self.motion_v_max),
            )
        logger.info("LoRA checkpoint loaded from %s", checkpoint_path)

    def clone_frozen(self) -> "ConanR1Model":
        """Return a frozen copy of this model for use as reference policy."""
        import copy
        ref = copy.deepcopy(self)
        for param in ref.model.parameters():
            param.requires_grad = False
        for module in (ref.reliability_pathway, ref.consistency_readouts):
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad = False
        return ref

    def _response_inputs(
        self,
        frames: List[np.ndarray],
        prompt: str,
        response: str,
        *,
        motion_frames: Optional[List[np.ndarray]] = None,
        elapsed_seconds: Optional[List[float]] = None,
        timestamps: Optional[List[float]] = None,
        reliability_intervention: str = "predicted",
    ):
        """Create multimodal inputs and labels masked to the response tokens."""
        pil_images = self._frames_to_pil(frames)
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in pil_images],
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        eos = self.processor.tokenizer.eos_token or ""
        full_text = text + response + eos
        prompt_inputs = self.processor(
            text=[text],
            images=pil_images or None,
            return_tensors="pt",
            padding=True,
        )
        inputs = self.processor(
            text=[full_text],
            images=pil_images or None,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels = inputs["input_ids"].clone()
        labels[:, :prompt_len] = -100
        if "attention_mask" in inputs:
            labels[inputs["attention_mask"] == 0] = -100
        inputs["labels"] = labels
        inputs, adapted = self._adapt_reliability_inputs(
            inputs,
            frames,
            motion_frames=motion_frames,
            elapsed_seconds=elapsed_seconds,
            timestamps=timestamps,
            reliability_intervention=reliability_intervention,
        )
        return inputs, adapted

    def response_nll(
        self,
        frames: List[np.ndarray],
        prompt: str,
        response: str,
        **visual_context,
    ) -> torch.Tensor:
        """Mean response-token NLL used for supervised fine-tuning."""
        inputs, _ = self._response_inputs(
            frames, prompt, response, **visual_context
        )
        outputs = self.model(**inputs)
        return outputs.loss

    def response_nll_with_state(
        self,
        frames: List[np.ndarray],
        prompt: str,
        response: str,
        require_diagnostic_slots: bool = True,
        **visual_context,
    ):
        """Return LM loss and the exact reliability state used by the decoder."""
        inputs, adapted = self._response_inputs(
            frames, prompt, response, **visual_context
        )
        if adapted is None:
            raise RuntimeError("The revised Stage-I objective requires reliability state.")
        outputs = self.model(
            **inputs, output_hidden_states=require_diagnostic_slots
        )
        if require_diagnostic_slots:
            self._attach_diagnostic_slots(adapted, outputs.hidden_states[-1])
        return outputs.loss, adapted

    def response_token_log_probs(
        self,
        frames: List[np.ndarray],
        prompt: str,
        response: str,
        require_grad: bool = True,
        **visual_context,
    ) -> torch.Tensor:
        """Return one conditional log probability for every response token.

        Prompt and visual tokens are excluded.  Keeping the token-level values
        is required for the clipped GRPO objective.
        """
        inputs, _ = self._response_inputs(
            frames, prompt, response, **visual_context
        )
        labels = inputs.pop("labels")
        context = nullcontext() if require_grad else torch.no_grad()
        forward_model = self.model if require_grad else self._policy_model()
        with context:
            outputs = forward_model(**inputs)
            shift_logits = outputs.logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            valid = shift_labels.ne(-100)
            safe_labels = shift_labels.masked_fill(~valid, 0)
            token_log_probs = -F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                safe_labels.reshape(-1),
                reduction="none",
            ).view_as(safe_labels)
            return token_log_probs[valid]

    def response_token_log_probs_with_state(
        self,
        frames: List[np.ndarray],
        prompt: str,
        response: str,
        require_grad: bool = True,
        require_diagnostic_slots: bool = True,
        **visual_context,
    ):
        """Return response-token log probabilities and the policy visual state."""
        inputs, adapted = self._response_inputs(
            frames, prompt, response, **visual_context
        )
        if adapted is None:
            raise RuntimeError("Full-policy GRPO requires the reliability pathway.")
        labels = inputs.pop("labels")
        context = nullcontext() if require_grad else torch.no_grad()
        forward_model = self.model if require_grad else self._policy_model()
        with context:
            outputs = forward_model(
                **inputs, output_hidden_states=require_diagnostic_slots
            )
            if require_diagnostic_slots:
                self._attach_diagnostic_slots(adapted, outputs.hidden_states[-1])
            shift_logits = outputs.logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            valid = shift_labels.ne(-100)
            safe_labels = shift_labels.masked_fill(~valid, 0)
            token_log_probs = -F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                safe_labels.reshape(-1),
                reduction="none",
            ).view_as(safe_labels)
        return token_log_probs[valid], adapted

    def _attach_diagnostic_slots(self, adapted, hidden_states: torch.Tensor) -> None:
        """Select decoder states immediately after the two opening tags."""
        for tag, attribute in (
            ("<TYPE>", "type_decoder_slot"),
            ("<INFLUENCE>", "influence_decoder_slot"),
        ):
            token_ids = self.processor.tokenizer(
                tag, add_special_tokens=False
            )["input_ids"]
            if not token_ids:
                raise RuntimeError(f"Tokenizer produced no ids for diagnostic tag {tag}.")
            sequence = adapted.input_ids[0].tolist()
            matches = [
                index
                for index in range(len(sequence) - len(token_ids) + 1)
                if sequence[index : index + len(token_ids)] == token_ids
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one {tag} marker in the supervised response, found {len(matches)}."
                )
            slot = min(matches[0] + len(token_ids), hidden_states.shape[1] - 1)
            setattr(adapted, attribute, hidden_states[:, slot, :])

    def log_prob(
        self,
        frames: List[np.ndarray],
        prompt: str,
        response: str,
        require_grad: bool = True,
        **visual_context,
    ) -> torch.Tensor:
        """Conditional sequence log probability retained for API compatibility."""
        return self.response_token_log_probs(
            frames,
            prompt,
            response,
            require_grad=require_grad,
            **visual_context,
        ).sum()
