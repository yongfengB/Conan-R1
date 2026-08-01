"""On-policy, token-level GRPO trainer for Conan-R1."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from model.conan_r1 import ConanR1Model
from model.parser import (
    extract_event_type,
    extract_temporal_interval,
    parse_structured_output,
)
from .grpo_math import clipped_grpo_loss, normalize_group_advantages
from .rewards import (
    canonicalize_factor,
    compute_rd,
    compute_re,
    compute_rl,
    compute_rt,
    compute_total_reward,
    validate_reward_weights,
)

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    lr: float = 1e-5
    weight_decay: float = 0.0
    epochs: int = 5
    group_size: int = 4
    update_epochs: int = 2
    clip_eps: float = 0.2
    kl_coef: float = 0.02
    logging_steps: int = 10
    save_steps: int = 200
    checkpoint_dir: str = "checkpoints/grpo"
    log_file: str = "training_log.jsonl"
    max_new_tokens: int = 384
    temperature: float = 0.9
    top_p: float = 0.95
    max_grad_norm: float = 1.0
    seed: int = 42
    # Reward weights
    w_d: float = 0.25
    w_e: float = 0.25
    w_t: float = 0.25
    w_l: float = 0.25
    # Degradation reward coefficients
    lambda_s: float = 0.5
    lambda_fp: float = 0.3
    lambda_fn: float = 0.3
    # Compactness reward
    length_tolerance: float = 0.20
    fixed_reasoning_target_length: Optional[int] = None
    # Structural ablation
    allow_missing_type_influence: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> "GRPOConfig":
        with open(path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        instance = cls()
        for section in ("training", "reward", "data", "output", "ablation"):
            for key, value in config.get(section, {}).items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
        validate_reward_weights(
            {
                "w_d": instance.w_d,
                "w_e": instance.w_e,
                "w_t": instance.w_t,
                "w_l": instance.w_l,
            }
        )
        if instance.group_size < 2:
            raise ValueError("group_size must be at least 2.")
        if instance.update_epochs < 1:
            raise ValueError("update_epochs must be at least 1.")
        return instance


@dataclass
class RewardBreakdown:
    rd: float
    re: float
    rt: float
    rl: float
    total: float
    parse_success: float


@dataclass
class CandidateRollout:
    response: str
    old_log_probs: torch.Tensor
    reference_log_probs: torch.Tensor
    advantage: torch.Tensor
    rewards: RewardBreakdown


class GRPOTrainer:
    """Align an SFT policy with four independently verifiable rewards.

    ``pi_ref`` is a fixed copy of the SFT policy used only for the KL term.
    ``pi_old`` is represented by token log probabilities stored immediately
    after each rollout and reused for ``update_epochs`` optimizer passes.
    """

    def __init__(
        self,
        model: ConanR1Model,
        ref_model: ConanR1Model,
        dataset,
        config: Optional[GRPOConfig] = None,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.dataset = dataset
        self.config = config or GRPOConfig()
        validate_reward_weights(
            {
                "w_d": self.config.w_d,
                "w_e": self.config.w_e,
                "w_t": self.config.w_t,
                "w_l": self.config.w_l,
            }
        )

        self.model.enable_gradient_checkpointing()
        self.model.disable_dropout()
        self.ref_model.disable_dropout()
        for parameter in self.ref_model.model.parameters():
            parameter.requires_grad = False
        self.ref_model.model.eval()

        trainable = [
            parameter
            for parameter in self.model.model.parameters()
            if parameter.requires_grad
        ]
        if not trainable:
            raise RuntimeError("GRPO policy has no trainable parameters.")
        self.optimizer = AdamW(
            trainable,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.optimizer_step = 0
        self.rollout_step = 0

    def sample_group(self, frames: List, prompt: str) -> List[str]:
        return [
            self.model.generate(
                frames,
                prompt,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            for _ in range(self.config.group_size)
        ]

    @staticmethod
    def _parse_profile(type_block: str) -> List:
        profile = []
        for entry in re_split_profile(type_block):
            if ":" not in entry:
                continue
            name, severity_text = entry.split(":", 1)
            try:
                profile.append(
                    (canonicalize_factor(name), float(severity_text.strip()))
                )
            except ValueError:
                continue
        return profile

    def _reward_breakdown(self, response: str, sample: dict) -> RewardBreakdown:
        optional_blocks = (
            ("TYPE", "INFLUENCE")
            if self.config.allow_missing_type_influence
            else ()
        )
        parsed = parse_structured_output(
            response, optional_blocks=optional_blocks
        )
        if parsed is None:
            return RewardBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        predicted_profile = self._parse_profile(parsed.type_block)
        ground_truth_profile = [
            (factor, severity)
            for factor, severity in sample.get("degradation_profile", [])
        ]
        rd = compute_rd(
            predicted_profile,
            ground_truth_profile,
            lambda_s=self.config.lambda_s,
            lambda_fp=self.config.lambda_fp,
            lambda_fn=self.config.lambda_fn,
        )

        predicted_event = extract_event_type(parsed.answer_block)
        event_aliases = sample.get("event_aliases", [])
        re_score = compute_re(
            predicted_event,
            sample.get("event_type", ""),
            aliases=event_aliases,
        )

        predicted_interval = extract_temporal_interval(parsed.answer_block)
        gt_interval = tuple(sample.get("gt_interval", [0.0, 1.0]))
        rt = compute_rt(
            predicted_interval,
            gt_interval,
            duration_sec=float(sample.get("duration_sec", 0.0)),
        )

        target_length = self.config.fixed_reasoning_target_length
        if target_length is None:
            target_length = int(sample.get("reasoning_target_length", 0))
        rl = compute_rl(
            parsed.reasoning_block,
            target_length,
            tolerance=self.config.length_tolerance,
        )
        total = compute_total_reward(
            rd,
            re_score,
            rt,
            rl,
            w_d=self.config.w_d,
            w_e=self.config.w_e,
            w_t=self.config.w_t,
            w_l=self.config.w_l,
        )
        return RewardBreakdown(rd, re_score, rt, rl, total, 1.0)

    def _collect_rollout(
        self, frames: List, prompt: str, sample: dict
    ) -> List[CandidateRollout]:
        """Sample responses and freeze their old/reference probabilities."""
        responses = self.sample_group(frames, prompt)
        reward_breakdowns = [
            self._reward_breakdown(response, sample) for response in responses
        ]
        reward_tensor = torch.tensor(
            [item.total for item in reward_breakdowns],
            dtype=torch.float32,
            device=self.model.device,
        )
        advantages = normalize_group_advantages(reward_tensor)

        candidates: List[CandidateRollout] = []
        for response, advantage, rewards in zip(
            responses, advantages, reward_breakdowns
        ):
            old_log_probs = self.model.response_token_log_probs(
                frames, prompt, response, require_grad=False
            ).detach()
            reference_log_probs = self.ref_model.response_token_log_probs(
                frames, prompt, response, require_grad=False
            ).detach()
            if old_log_probs.numel() == 0:
                continue
            if old_log_probs.shape != reference_log_probs.shape:
                raise RuntimeError(
                    "Policy and reference tokenization produced different lengths."
                )
            candidates.append(
                CandidateRollout(
                    response=response,
                    old_log_probs=old_log_probs,
                    reference_log_probs=reference_log_probs,
                    advantage=advantage.detach(),
                    rewards=rewards,
                )
            )
        if not candidates:
            raise RuntimeError("No response tokens were generated for this rollout.")
        return candidates

    def _optimize_rollout(
        self, frames: List, prompt: str, candidates: List[CandidateRollout]
    ) -> Dict[str, float]:
        diagnostic_rows = []
        losses = []
        for _ in range(self.config.update_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            update_losses = []
            for candidate in candidates:
                current_log_probs = self.model.response_token_log_probs(
                    frames,
                    prompt,
                    candidate.response,
                    require_grad=True,
                )
                loss, diagnostics = clipped_grpo_loss(
                    current_log_probs=current_log_probs,
                    old_log_probs=candidate.old_log_probs,
                    reference_log_probs=candidate.reference_log_probs,
                    advantage=candidate.advantage,
                    clip_eps=self.config.clip_eps,
                    kl_coef=self.config.kl_coef,
                )
                (loss / len(candidates)).backward()
                update_losses.append(loss.detach())
                diagnostic_rows.append(diagnostics)

            torch.nn.utils.clip_grad_norm_(
                self.model.model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer_step += 1
            losses.extend(update_losses)

        def mean_tensor(key: str) -> float:
            return float(
                torch.stack([row[key].float() for row in diagnostic_rows])
                .mean()
                .cpu()
            )

        return {
            "loss": float(torch.stack(losses).mean().cpu()),
            "ratio_mean": mean_tensor("ratio_mean"),
            "clip_fraction": mean_tensor("clip_fraction"),
            "approx_kl": mean_tensor("approx_kl"),
            "policy_objective": mean_tensor("policy_objective"),
        }

    def _log_record(self, record: Dict) -> None:
        if not is_main_process():
            return
        output = Path(self.config.checkpoint_dir)
        output.mkdir(parents=True, exist_ok=True)
        with open(output / self.config.log_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def train(self) -> None:
        generator = torch.Generator()
        generator.manual_seed(self.config.seed)
        sampler = (
            DistributedSampler(
                self.dataset,
                shuffle=True,
                seed=self.config.seed,
                drop_last=False,
            )
            if dist.is_available() and dist.is_initialized()
            else None
        )
        loader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=0,
            collate_fn=lambda samples: samples[0],
            generator=generator,
        )
        self.model.model.train()

        for epoch in range(self.config.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            for sample in tqdm(
                loader, desc=f"GRPO Epoch {epoch + 1}/{self.config.epochs}"
            ):
                frames_tensor = sample["frames"]
                frame_array = (
                    frames_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255
                ).astype("uint8")
                frames = [
                    frame_array[index] for index in range(frame_array.shape[0])
                ]
                prompt = sample["prompt"]

                candidates = self._collect_rollout(frames, prompt, sample)
                optimization = self._optimize_rollout(
                    frames, prompt, candidates
                )
                self.rollout_step += 1
                components = {
                    key: sum(
                        getattr(candidate.rewards, key)
                        for candidate in candidates
                    )
                    / len(candidates)
                    for key in ("rd", "re", "rt", "rl", "total", "parse_success")
                }
                record = {
                    "epoch": epoch + 1,
                    "rollout_step": self.rollout_step,
                    "optimizer_step": self.optimizer_step,
                    "video_id": sample.get("video_id", ""),
                    **optimization,
                    **{f"reward_{key}": value for key, value in components.items()},
                }
                if self.rollout_step % self.config.logging_steps == 0:
                    logger.info(
                        "rollout=%d loss=%.4f reward=%.4f clip=%.4f kl=%.5f",
                        self.rollout_step,
                        record["loss"],
                        record["reward_total"],
                        record["clip_fraction"],
                        record["approx_kl"],
                    )
                self._log_record(record)

                if (
                    self.optimizer_step > 0
                    and self.optimizer_step % self.config.save_steps == 0
                ):
                    self._save(
                        os.path.join(
                            self.config.checkpoint_dir,
                            f"step_{self.optimizer_step}",
                        )
                    )

        self._save(self.config.checkpoint_dir)
        logger.info(
            "GRPO complete: %d rollouts, %d optimizer steps.",
            self.rollout_step,
            self.optimizer_step,
        )

    def _save(self, path: str) -> None:
        if not is_main_process():
            return
        os.makedirs(path, exist_ok=True)
        self.model.save_lora(path)
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
        with open(
            os.path.join(path, "trainer_state.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(
                {
                    "rollout_step": self.rollout_step,
                    "optimizer_step": self.optimizer_step,
                    "config": asdict(self.config),
                },
                handle,
                indent=2,
            )


def is_main_process() -> bool:
    return (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_rank() == 0
    )


def re_split_profile(type_block: str) -> List[str]:
    """Split a TYPE block while tolerating commas between factor entries."""
    normalized = type_block.replace("\n", ";")
    return [
        entry.strip()
        for entry in normalized.replace(",", ";").split(";")
        if entry.strip()
    ]
