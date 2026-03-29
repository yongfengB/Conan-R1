"""GRPO (Group Relative Policy Optimization) trainer for Conan-R1."""
from __future__ import annotations
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.conan_r1 import ConanR1Model
from model.parser import parse_structured_output, extract_temporal_interval
from .rewards import compute_ro, compute_rt, compute_rl, compute_total_reward

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    lr: float = 1e-5
    epochs: int = 5
    group_size: int = 4
    clip_eps: float = 0.2
    kl_coef: float = 0.02
    logging_steps: int = 50
    save_steps: int = 200
    checkpoint_dir: str = "checkpoints/grpo"
    # Reward weights
    alpha: float = 0.4
    beta: float = 0.4
    gamma: float = 0.2
    # Reward coefficients
    lambda_s: float = 0.5
    lambda_fp: float = 0.3
    lambda_fn: float = 0.3

    @classmethod
    def from_yaml(cls, path: str) -> "GRPOConfig":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        obj = cls()
        for section in ("training", "reward", "output"):
            for k, v in cfg.get(section, {}).items():
                if hasattr(obj, k):
                    setattr(obj, k, v)
        if "checkpoint_dir" in cfg.get("output", {}):
            obj.checkpoint_dir = cfg["output"]["checkpoint_dir"]
        return obj


class GRPOTrainer:
    """GRPO trainer: aligns Conan-R1 with ro, rt, rl rewards.

    Initializes from an SFT checkpoint and maintains a frozen reference policy.
    """

    def __init__(
        self,
        model: ConanR1Model,
        ref_model: ConanR1Model,
        dataset,
        config: Optional[GRPOConfig] = None,
    ) -> None:
        self.model = model
        self.ref_model = ref_model  # frozen reference policy
        self.dataset = dataset
        self.config = config or GRPOConfig()

        # Freeze reference model
        for param in self.ref_model.model.parameters():
            param.requires_grad = False

        self.optimizer = AdamW(
            [p for p in self.model.model.parameters() if p.requires_grad],
            lr=self.config.lr,
        )

    # ------------------------------------------------------------------
    # Group sampling
    # ------------------------------------------------------------------

    def sample_group(
        self,
        frames_list: List,
        prompt: str,
        G: int = 4,
    ) -> List[str]:
        """Sample G candidate responses from the current policy."""
        responses = []
        for _ in range(G):
            resp = self.model.generate(frames_list, prompt)
            responses.append(resp)
        return responses

    # ------------------------------------------------------------------
    # Advantage computation
    # ------------------------------------------------------------------

    def compute_advantages(self, rewards: List[float]) -> List[float]:
        """Normalize group-relative advantages.

        Adv_i = (R_i - mean(R)) / (std(R) + 1e-6)
        """
        arr = np.array(rewards, dtype=np.float64)
        mean = arr.mean()
        std = arr.std() + 1e-6
        return ((arr - mean) / std).tolist()

    # ------------------------------------------------------------------
    # Reward computation for a single response
    # ------------------------------------------------------------------

    def _compute_reward(self, response: str, sample: dict) -> float:
        parsed = parse_structured_output(response)
        if parsed is None:
            return 0.0

        # ro
        pred_profile = []
        for entry in parsed.type_block.split(";"):
            entry = entry.strip()
            if ":" in entry:
                name, sev_str = entry.split(":", 1)
                try:
                    pred_profile.append((name.strip(), float(sev_str.strip())))
                except ValueError:
                    pass
        gt_profile = [(f, s) for f, s in sample.get("degradation_profile", [])]
        ro = compute_ro(pred_profile, gt_profile,
                        self.config.lambda_s, self.config.lambda_fp, self.config.lambda_fn)

        # rt
        pred_interval = extract_temporal_interval(parsed.answer_block)
        gt_interval = tuple(sample.get("gt_interval", [0.0, 1.0]))
        rt = compute_rt(pred_interval, gt_interval)

        # rl
        gt_reasoning = sample.get("reasoning_annotation", "")
        rl = compute_rl(parsed.reasoning_block, gt_reasoning)

        return compute_total_reward(ro, rt, rl, self.config.alpha, self.config.beta, self.config.gamma)

    # ------------------------------------------------------------------
    # GRPO policy update step
    # ------------------------------------------------------------------

    def _grpo_step(self, frames_list: List, prompt: str, sample: dict) -> torch.Tensor:
        G = self.config.group_size
        responses = self.sample_group(frames_list, prompt, G)

        rewards = [self._compute_reward(r, sample) for r in responses]
        advantages = self.compute_advantages(rewards)

        total_loss = torch.tensor(0.0, device=self.model.device, requires_grad=True)

        for resp, adv in zip(responses, advantages):
            # Current policy log prob
            log_p_curr = self.model.log_prob(frames_list, prompt, resp)
            # Reference policy log prob (no grad)
            with torch.no_grad():
                log_p_ref = self.ref_model.log_prob(frames_list, prompt, resp)

            # Importance ratio ρ = exp(log π_θ - log π_old)
            # Here we use ref as old policy (simplified)
            rho = torch.exp(log_p_curr - log_p_ref.detach())

            adv_t = torch.tensor(adv, dtype=torch.float32, device=self.model.device)
            clipped = torch.clamp(rho, 1 - self.config.clip_eps, 1 + self.config.clip_eps)
            pg_loss = -torch.min(rho * adv_t, clipped * adv_t)

            # KL penalty: KL(π_θ || π_ref) ≈ log_p_curr - log_p_ref
            kl = log_p_curr - log_p_ref.detach()
            step_loss = pg_loss + self.config.kl_coef * kl
            total_loss = total_loss + step_loss / G

        return total_loss

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        loader = DataLoader(self.dataset, batch_size=1, shuffle=True, num_workers=0)
        self.model.model.train()
        global_step = 0

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            for batch in tqdm(loader, desc=f"GRPO Epoch {epoch + 1}/{self.config.epochs}"):
                frames_tensor = batch["frames"][0]  # (T, C, H, W)
                frames_np = (frames_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")
                frames_list = [frames_np[t] for t in range(frames_np.shape[0])]
                prompt = batch["prompt"][0]
                sample = {k: v[0] if isinstance(v, list) else v for k, v in batch.items()}

                self.optimizer.zero_grad()
                loss = self._grpo_step(frames_list, prompt, sample)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                global_step += 1

                if global_step % self.config.logging_steps == 0:
                    logger.info("GRPO Step %d | loss=%.4f", global_step, loss.item())

                if global_step % self.config.save_steps == 0:
                    ckpt = os.path.join(self.config.checkpoint_dir, f"step_{global_step}")
                    self._save(ckpt)

            avg = epoch_loss / max(1, len(loader))
            logger.info("GRPO Epoch %d | avg_loss=%.4f", epoch + 1, avg)

        self._save(self.config.checkpoint_dir)
        logger.info("GRPO training complete. Saved to %s", self.config.checkpoint_dir)

    def _save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save_lora(path)
