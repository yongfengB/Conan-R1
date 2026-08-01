"""Supervised Fine-Tuning trainer for Conan-R1."""
from __future__ import annotations
import logging
import os
import json
from dataclasses import asdict, dataclass, field
import math
from typing import Optional

import torch
import torch.distributed as dist
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import get_linear_schedule_with_warmup
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
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    epochs: int = 10
    warmup_steps: int = 100
    seed: int = 42
    # Logging & checkpointing
    logging_steps: int = 50
    save_steps: int = 500
    checkpoint_dir: str = "checkpoints/sft"
    log_file: str = "training_log.jsonl"
    # Data
    max_new_tokens: int = 384
    enabled_blocks: list = field(
        default_factory=lambda: [
            "TYPE",
            "INFLUENCE",
            "REASONING",
            "CONCLUSION",
            "ANSWER",
        ]
    )

    @classmethod
    def from_yaml(cls, path: str) -> "SFTConfig":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        obj = cls()
        train_cfg = cfg.get("training", {})
        for k, v in train_cfg.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        for k, v in cfg.get("data", {}).items():
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
        self.model.enable_gradient_checkpointing()

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
        self.scheduler = None
        self.global_step = 0

    def _log_record(self, record: dict) -> None:
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        output = os.path.abspath(self.config.checkpoint_dir)
        os.makedirs(output, exist_ok=True)
        with open(
            os.path.join(output, self.config.log_file), "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _build_target_sequence(self, sample: dict) -> str:
        """Serialize a sample into the structured target sequence Y."""
        values = {
            "TYPE": sample["type_annotation"],
            "INFLUENCE": sample["influence_annotation"],
            "REASONING": sample["reasoning_annotation"],
            "CONCLUSION": sample["conclusion_annotation"],
            "ANSWER": sample["answer_annotation"],
        }
        enabled = [block.upper() for block in self.config.enabled_blocks]
        unknown = set(enabled) - set(values)
        if unknown:
            raise ValueError(f"Unknown structured blocks: {sorted(unknown)}")
        if "ANSWER" not in enabled:
            raise ValueError("ANSWER must remain enabled for task supervision.")
        return "".join(
            f"<{block}>{values[block]}<{block}_END>" for block in enabled
        )

    def _compute_loss(self, batch: list[dict]) -> torch.Tensor:
        """Compute cross-entropy loss on the structured target sequence."""
        losses = []
        for sample in batch:
            frames_tensor = sample["frames"]  # (T, C, H, W)
            frames_np = (frames_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")
            frames_list = [frames_np[t] for t in range(frames_np.shape[0])]

            target = self._build_target_sequence(sample)
            losses.append(
                self.model.response_nll(frames_list, sample["prompt"], target)
            )

        return torch.stack(losses).mean()

    def train(self) -> None:
        """Run SFT training loop."""
        generator = torch.Generator()
        generator.manual_seed(self.config.seed)
        sampler = (
            DistributedSampler(
                self.dataset, shuffle=True, seed=self.config.seed, drop_last=False
            )
            if dist.is_available() and dist.is_initialized()
            else None
        )
        loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=0,
            collate_fn=lambda samples: samples,
            generator=generator,
        )
        updates_per_epoch = math.ceil(
            len(loader) / self.config.gradient_accumulation_steps
        )
        total_updates = max(1, updates_per_epoch * self.config.epochs)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=min(self.config.warmup_steps, total_updates),
            num_training_steps=total_updates,
        )
        self.model.model.train()
        global_step = 0
        self.optimizer.zero_grad(set_to_none=True)

        for epoch in range(self.config.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(
                tqdm(loader, desc=f"SFT Epoch {epoch + 1}/{self.config.epochs}")
            ):
                loss = self._compute_loss(batch)
                (loss / self.config.gradient_accumulation_steps).backward()
                epoch_loss += loss.item()

                should_update = (
                    (batch_idx + 1) % self.config.gradient_accumulation_steps == 0
                    or batch_idx + 1 == len(loader)
                )
                if not should_update:
                    continue

                torch.nn.utils.clip_grad_norm_(
                    self.model.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                global_step += 1
                self.global_step = global_step
                self._log_record(
                    {
                        "epoch": epoch + 1,
                        "optimizer_step": global_step,
                        "loss": float(loss.item()),
                        "learning_rate": float(
                            self.scheduler.get_last_lr()[0]
                        ),
                    }
                )

                if global_step % self.config.logging_steps == 0:
                    logger.info(
                        "Update %d | loss=%.4f | lr=%.3e",
                        global_step,
                        loss.item(),
                        self.scheduler.get_last_lr()[0],
                    )

                if global_step % self.config.save_steps == 0:
                    ckpt = os.path.join(self.config.checkpoint_dir, f"step_{global_step}")
                    self.save_checkpoint(ckpt)

            avg = epoch_loss / max(1, len(loader))
            logger.info("Epoch %d complete | avg_loss=%.4f", epoch + 1, avg)

        self.save_checkpoint(self.config.checkpoint_dir)
        logger.info("SFT training complete. Checkpoint saved to %s", self.config.checkpoint_dir)

    def save_checkpoint(self, path: str) -> None:
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        os.makedirs(path, exist_ok=True)
        self.model.save_lora(path)
        torch.save(
            self.optimizer.state_dict(), os.path.join(path, "optimizer.pt")
        )
        if self.scheduler is not None:
            torch.save(
                self.scheduler.state_dict(), os.path.join(path, "scheduler.pt")
            )
        with open(
            os.path.join(path, "trainer_state.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(
                {
                    "optimizer_step": self.global_step,
                    "config": asdict(self.config),
                },
                handle,
                indent=2,
            )
        logger.info("Checkpoint saved: %s", path)
