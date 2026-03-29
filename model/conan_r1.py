"""ConanR1Model: Qwen2.5-VL-3B + LoRA for structured anomaly understanding."""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

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
    target_modules: Optional[List[str]] = None  # None → auto-detect

    def to_peft_config(self) -> LoraConfig:
        kwargs = dict(
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        if self.target_modules:
            kwargs["target_modules"] = self.target_modules
        return LoraConfig(**kwargs)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ConanR1Model:
    """Qwen2.5-VL-3B with LoRA adapters for structured five-block generation."""

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-VL-3B",
        lora_config: Optional[LoRAConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.base_model_name = base_model
        self.lora_config = lora_config or LoRAConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading base model: %s", base_model)
        self.processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model = get_peft_model(self.model, self.lora_config.to_peft_config())
        self.model.to(self.device)
        logger.info("Model loaded on %s with LoRA (rank=%d, alpha=%d).",
                    self.device, self.lora_config.rank, self.lora_config.alpha)

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

        # Build conversation format expected by Qwen2-VL
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
            images=pil_images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy decoding
            )

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

    # ------------------------------------------------------------------
    # LoRA checkpoint management
    # ------------------------------------------------------------------

    def save_lora(self, checkpoint_path: str) -> None:
        """Save only the LoRA adapter weights."""
        self.model.save_pretrained(checkpoint_path)
        self.processor.save_pretrained(checkpoint_path)
        logger.info("LoRA checkpoint saved to %s", checkpoint_path)

    def load_lora(self, checkpoint_path: str) -> None:
        """Load LoRA adapter weights from a checkpoint."""
        self.model = PeftModel.from_pretrained(
            self.model.base_model.model,
            checkpoint_path,
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

    def log_prob(
        self,
        frames: List[np.ndarray],
        prompt: str,
        response: str,
    ) -> torch.Tensor:
        """Compute the log probability of `response` given (frames, prompt).

        Used by GRPO to compute the importance ratio ρ_i.
        """
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
        full_text = text + response
        inputs = self.processor(
            text=[full_text],
            images=pil_images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
        return -outputs.loss  # negative NLL ≈ log prob
