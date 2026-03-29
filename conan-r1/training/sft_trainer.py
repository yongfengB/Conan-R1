"""Supervised Fine-Tuning trainer for Conan-R1."""
from __future__ import annotations
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.conan_r1 import ConanR1Model

logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    # Optimizer
    lr: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    # Training loop
    batch_size: int = 16
    epochs: int = 10
    warmup_steps: int = 100
    # Logging & checkpointing
    logging_steps: int = 50
    save_steps: int = 500
    checkpoint_dir: str = "checkpoints/sft"
    # Data
    max_new_tokens: int = 384

    @classmethod
    def from_yaml(cls, path: str) -> "SFTConfig":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        obj = cls()
        train_cfg = cfg.get("training", {})
        for k, v in train_cfg.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        out_cfg = cfg.get("output", {})
        if "checkpoint_dir" in out_cfg:
            obj.checkpoint_dir = out_cfg["checkpoint_dir"]
        return obj


class SFTTrainer:
    """Trains ConanR1Model with cross-entropy loss on structured sequences.

    Only LoRA adapter parameters are updated; backbone weights are frozen.
    """

    def __init__(
        self,
        model: ConanR1Model,
        dataset,
        config: Optional[SFTConfig] = None,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.config = config or SFTConfig()

        # Freeze backbone, only train LoRA params
        for name, param in self.model.model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.model.parameters())
        logger.info("Trainable params: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total)

        self.optimizer = AdamW(
            [p for p in self.model.model.parameters() if p.requires_grad],
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    def _build_target_sequence(self, sample: dict) -> str:
        """Serialize a sample into the structured target sequence Y."""
        return (
            f"<TYPE>{sample['type_annotation']}<TYPE_END>"
            f"<INFLUENCE>{sample['influence_annotation']}<INFLUENCE_END>"
            f"<REASONING>{sample['reasoning_annotation']}<REASONING_END>"
            f"<CONCLUSION>{sample['conclusion_annotation']}<CONCLUSION_END>"
            f"<ANSWER>{sample['answer_annotation']}<ANSWER_END>"
        )

    def _compute_loss(self, batch: dict) -> torch.Tensor:
        """Compute cross-entropy loss on the structured target sequence."""
        losses = []
        for i in range(len(batch["video_id"])):
            frames_tensor = batch["frames"][i]  # (T, C, H, W)
            frames_np = (frames_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")
            frames_list = [frames_np[t] for t in range(frames_np.shape[0])]

            prompt = batch["prompt"][i]
            target = self._build_target_sequence({
                "type_annotation": batch["type_annotation"][i],
                "influence_annotation": batch["influence_annotation"][i],
                "reasoning_annotation": batch["reasoning_annotation"][i],
                "conclusion_annotation": batch["conclusion_annotation"][i],
                "answer_annotation": batch["answer_annotation"][i],
            })

            # Use model's log_prob as a proxy for cross-entropy loss
            log_p = self.model.log_prob(frames_list, prompt, target)
            losses.append(-log_p)

        return torch.stack(losses).mean()

    def train(self) -> None:
        """Run SFT training loop."""
        loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        self.model.model.train()
        global_step = 0

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            for batch in tqdm(loader, desc=f"SFT Epoch {epoch + 1}/{self.config.epochs}"):
                self.optimizer.zero_grad()
                loss = self._compute_loss(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                epoch_loss += loss.item()
                global_step += 1

                if global_step % self.config.logging_steps == 0:
                    logger.info("Step %d | loss=%.4f", global_step, loss.item())

                if global_step % self.config.save_steps == 0:
                    ckpt = os.path.join(self.config.checkpoint_dir, f"step_{global_step}")
                    self.save_checkpoint(ckpt)

            avg = epoch_loss / max(1, len(loader))
            logger.info("Epoch %d complete | avg_loss=%.4f", epoch + 1, avg)

        self.save_checkpoint(self.config.checkpoint_dir)
        logger.info("SFT training complete. Checkpoint saved to %s", self.config.checkpoint_dir)

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save_lora(path)
        logger.info("Checkpoint saved: %s", path)
