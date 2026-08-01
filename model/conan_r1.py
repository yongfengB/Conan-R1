"""Qwen2.5-VL-3B-Instruct policy used by Conan-R1."""
from __future__ import annotations
from contextlib import nullcontext
import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .parser import StructuredOutput, parse_structured_output

logger = logging.getLogger(__name__)


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
        lora_config: Optional[LoRAConfig] = None,
        device: Optional[str] = None,
        enable_lora: bool = True,
    ) -> None:
        self.base_model_name = base_model
        self.lora_config = lora_config or LoRAConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.startswith("cuda") else torch.float32

        logger.info("Loading base model: %s", base_model)
        self.processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )
        self.enable_lora = enable_lora
        if enable_lora:
            self.model = get_peft_model(
                self.model, self.lora_config.to_peft_config()
            )
        self.model.to(self.device)
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
        generated = output_ids[:, input_len:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0]

    def generate_structured(
        self,
        frames: List[np.ndarray],
        prompt: str,
        max_new_tokens: int = 384,
    ) -> Optional[StructuredOutput]:
        """Generate and parse a structured output.

        Returns None if the output cannot be parsed.
        """
        raw = self.generate(frames, prompt, max_new_tokens)
        return parse_structured_output(raw)

    def generate_with_prefix(
        self,
        frames: List[np.ndarray],
        prompt: str,
        response_prefix: str,
        max_new_tokens: int = 384,
    ) -> str:
        """Greedily continue a forced structured prefix for intervention tests."""
        pil_images = self._frames_to_pil(frames)
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": image} for image in pil_images],
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_prompt = prompt_text + response_prefix
        inputs = self.processor(
            text=[full_prompt],
            images=pil_images or None,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        policy_model = self._policy_model()
        was_training = policy_model.training
        policy_model.eval()
        with torch.inference_mode():
            output_ids = policy_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        if was_training:
            policy_model.train()
        input_length = inputs["input_ids"].shape[1]
        continuation = self.processor.batch_decode(
            output_ids[:, input_length:], skip_special_tokens=True
        )[0]
        return response_prefix + continuation

    # ------------------------------------------------------------------
    # LoRA checkpoint management
    # ------------------------------------------------------------------

    def save_lora(self, checkpoint_path: str) -> None:
        """Save only the LoRA adapter weights."""
        if not self.enable_lora:
            raise RuntimeError("Cannot save a LoRA adapter when LoRA is disabled.")
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
        logger.info("LoRA checkpoint loaded from %s", checkpoint_path)

    def clone_frozen(self) -> "ConanR1Model":
        """Return a frozen copy of this model for use as reference policy."""
        import copy
        ref = copy.deepcopy(self)
        for param in ref.model.parameters():
            param.requires_grad = False
        return ref

    def _response_inputs(
        self,
        frames: List[np.ndarray],
        prompt: str,
        response: str,
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
        return inputs

    def response_nll(
        self,
        frames: List[np.ndarray],
        prompt: str,
        response: str,
    ) -> torch.Tensor:
        """Mean response-token NLL used for supervised fine-tuning."""
        inputs = self._response_inputs(frames, prompt, response)
        outputs = self.model(**inputs)
        return outputs.loss

    def response_token_log_probs(
        self,
        frames: List[np.ndarray],
        prompt: str,
        response: str,
        require_grad: bool = True,
    ) -> torch.Tensor:
        """Return one conditional log probability for every response token.

        Prompt and visual tokens are excluded.  Keeping the token-level values
        is required for the clipped GRPO objective.
        """
        inputs = self._response_inputs(frames, prompt, response)
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

    def log_prob(
        self,
        frames: List[np.ndarray],
        prompt: str,
        response: str,
        require_grad: bool = True,
    ) -> torch.Tensor:
        """Conditional sequence log probability retained for API compatibility."""
        return self.response_token_log_probs(
            frames, prompt, response, require_grad=require_grad
        ).sum()
