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
    extract_degradation_profile,
    parse_answer_fields,
    parse_structured_output,
)
from .grpo_math import clipped_grpo_loss, normalize_group_advantages
from .auxiliary import encode_degradation_profile, rasterize_logged_occlusions
from .stage_objectives import AuxiliaryLossWeights, stage2_loss
from .precision import make_grad_scaler, require_finite
from .rewards import (
    compute_rd,
    compute_re,
    compute_rl,
    compute_rt,
    compute_task_masked_reward,
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
    # One-sided compactness reward
    compactness_base_budget: int = 32
    compactness_per_task_budget: int = 32
    # Structural ablation
    allow_missing_type_influence: bool = False
    policy_scope: str = "full"
    pathway_mode: str = "reliability"
    task_masking: bool = True
    preserve_during_grpo: bool = True
    lambda_d_rl: float = 1.0
    lambda_q_rl: float = 1.0
    lambda_c_rl: float = 0.1
    reliability_target: str = "ema"
    occlusion_mask_adjustment: bool = True
    ema_update_after_optimizer_step: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "GRPOConfig":
        with open(path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        instance = cls()
        for section in (
            "training",
            "reward",
            "data",
            "output",
            "ablation",
            "auxiliary_losses",
            "model",
        ):
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
        if instance.policy_scope not in {"full", "lora_only"}:
            raise ValueError("policy_scope must be full or lora_only.")
        if instance.pathway_mode not in {"reliability", "none"}:
            raise ValueError("pathway_mode must be reliability or none.")
        if instance.pathway_mode == "none" and instance.policy_scope != "lora_only":
            raise ValueError("A policy without the reliability pathway is LoRA-only.")
        if instance.lr <= 0.0 or instance.epochs < 1:
            raise ValueError("GRPO lr and epochs must be positive.")
        if not 0.0 < instance.clip_eps < 1.0 or instance.kl_coef < 0.0:
            raise ValueError("clip_eps must be in (0, 1) and kl_coef non-negative.")
        if instance.temperature <= 0.0 or not 0.0 < instance.top_p <= 1.0:
            raise ValueError("temperature and top_p must define valid sampling.")
        if instance.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive.")
        if instance.logging_steps < 1 or instance.save_steps < 1:
            raise ValueError("logging_steps and save_steps must be positive.")
        if min(instance.lambda_d_rl, instance.lambda_q_rl, instance.lambda_c_rl) < 0.0:
            raise ValueError("Auxiliary loss weights must be non-negative.")
        if instance.allow_missing_type_influence and instance.lambda_c_rl > 0.0:
            raise ValueError(
                "lambda_c_rl must be zero when TYPE or INFLUENCE may be absent."
            )
        if instance.preserve_during_grpo and (
            instance.pathway_mode == "none" or instance.policy_scope != "full"
        ):
            raise ValueError(
                "Auxiliary preservation requires the full reliability policy."
            )
        target_aliases = {
            "frozen_appearance_teacher_plus_ema_motion_teacher": "ema",
            "ema": "ema",
            "online_motion_target": "online",
            "online": "online",
            "frozen_motion_teacher": "frozen_initial",
            "frozen_initial": "frozen_initial",
        }
        if instance.reliability_target not in target_aliases:
            raise ValueError("Unsupported reliability_target.")
        instance.reliability_target = target_aliases[instance.reliability_target]
        if (
            instance.reliability_target == "frozen_initial"
            and instance.ema_update_after_optimizer_step
        ):
            raise ValueError("A frozen motion teacher cannot receive EMA updates.")
        return instance


@dataclass
class RewardBreakdown:
    rd: float
    re: float
    rt: float
    rl: float
    total: float
    parse_success: float
    event_active: float = 1.0
    temporal_active: float = 1.0
    active_fields_valid: float = 1.0


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
        if self.config.policy_scope == "lora_only":
            for module in (
                self.model.reliability_pathway,
                self.model.consistency_readouts,
                self.model.motion_teacher,
            ):
                if module is None:
                    continue
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        for module in (
            self.ref_model.model,
            self.ref_model.reliability_pathway,
            self.ref_model.consistency_readouts,
            self.ref_model.motion_teacher,
        ):
            if module is None:
                continue
            for parameter in module.parameters():
                parameter.requires_grad_(False)
            module.eval()

        trainable = [
            parameter
            for _, parameter in self.model.trainable_policy_named_parameters(
                self.config.policy_scope
            )
        ]
        if not trainable:
            raise RuntimeError("GRPO policy has no trainable parameters.")
        self.optimizer = AdamW(
            trainable,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.scaler = make_grad_scaler(self.model.device, self.model.dtype)
        self.optimizer_step = 0
        self.rollout_step = 0

    def sample_group(
        self, frames: List, prompt: str, visual_context: dict
    ) -> List[str]:
        return [
            self.model.generate(
                frames,
                prompt,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                **visual_context,
            )
            for _ in range(self.config.group_size)
        ]

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

        predicted_profile = (
            []
            if self.config.allow_missing_type_influence and not parsed.type_block
            else extract_degradation_profile(parsed.type_block)
        )
        if predicted_profile is None:
            return RewardBreakdown(
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                active_fields_valid=0.0,
            )
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

        event_active = bool(sample.get("task_mask", {}).get("event", True))
        temporal_active = bool(sample.get("task_mask", {}).get("temporal", True))
        if not self.config.task_masking:
            event_active = temporal_active = True
        answer_fields = parse_answer_fields(
            parsed.answer_block,
            event_active=event_active,
            temporal_active=temporal_active,
            duration_sec=float(sample.get("duration_sec", 0.0)),
        )
        if answer_fields is None:
            return RewardBreakdown(
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                float(event_active),
                float(temporal_active),
                0.0,
            )
        predicted_event = answer_fields.event_type
        event_aliases = sample.get("event_aliases", [])
        re_score = compute_re(
            predicted_event,
            sample.get("event_type", ""),
            aliases=event_aliases,
        )

        predicted_interval = answer_fields.interval
        gt_interval = tuple(sample.get("gt_interval", [0.0, 1.0]))
        rt = compute_rt(
            predicted_interval,
            gt_interval,
            duration_sec=float(sample.get("duration_sec", 0.0)),
        )

        rl = compute_rl(
            parsed.reasoning_block,
            event_active=event_active,
            temporal_active=temporal_active,
            base_budget=self.config.compactness_base_budget,
            per_task_budget=self.config.compactness_per_task_budget,
        )
        active_fields_valid = True
        total = compute_task_masked_reward(
            rd,
            re_score,
            rt,
            rl,
            event_active=event_active,
            temporal_active=temporal_active,
            active_fields_valid=active_fields_valid,
            w_d=self.config.w_d,
            w_e=self.config.w_e,
            w_t=self.config.w_t,
            w_l=self.config.w_l,
        )
        if not active_fields_valid:
            rd = re_score = rt = rl = 0.0
        return RewardBreakdown(
            rd,
            re_score,
            rt,
            rl,
            total,
            1.0,
            float(event_active),
            float(temporal_active),
            float(active_fields_valid),
        )

    def _collect_rollout(
        self, frames: List, prompt: str, sample: dict, visual_context: dict
    ) -> List[CandidateRollout]:
        """Sample responses and freeze their old/reference probabilities."""
        responses = self.sample_group(frames, prompt, visual_context)
        reward_breakdowns = [
            self._reward_breakdown(response, sample) for response in responses
        ]
        retained = []
        for response, rewards in zip(responses, reward_breakdowns):
            old_log_probs = self.model.response_token_log_probs(
                frames, prompt, response, require_grad=False, **visual_context
            ).detach()
            reference_log_probs = self.ref_model.response_token_log_probs(
                frames, prompt, response, require_grad=False, **visual_context
            ).detach()
            if old_log_probs.numel() == 0:
                continue
            if old_log_probs.shape != reference_log_probs.shape:
                raise RuntimeError(
                    "Policy and reference tokenization produced different lengths."
                )
            retained.append((response, rewards, old_log_probs, reference_log_probs))
        if not retained:
            raise RuntimeError("No response tokens were generated for this rollout.")
        reward_tensor = torch.tensor(
            [item[1].total for item in retained],
            dtype=torch.float32,
            device=self.model.device,
        )
        advantages = normalize_group_advantages(reward_tensor)
        candidates: List[CandidateRollout] = []
        for (response, rewards, old_log_probs, reference_log_probs), advantage in zip(
            retained, advantages
        ):
            candidates.append(
                CandidateRollout(
                    response=response,
                    old_log_probs=old_log_probs,
                    reference_log_probs=reference_log_probs,
                    advantage=advantage.detach(),
                    rewards=rewards,
                )
            )
        return candidates

    def _optimize_rollout(
        self,
        frames: List,
        prompt: str,
        candidates: List[CandidateRollout],
        visual_context: dict,
        sample: dict,
    ) -> Dict[str, float]:
        diagnostic_rows = []
        losses = []
        for _ in range(self.config.update_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            update_losses = []
            for candidate in candidates:
                if self.model.reliability_pathway is None:
                    current_log_probs = self.model.response_token_log_probs(
                        frames,
                        prompt,
                        candidate.response,
                        require_grad=True,
                    )
                    degraded_state = None
                else:
                    current_log_probs, degraded_state = (
                        self.model.response_token_log_probs_with_state(
                            frames,
                            prompt,
                            candidate.response,
                            require_grad=True,
                            require_diagnostic_slots=(
                                self.config.preserve_during_grpo
                                and self.config.lambda_c_rl > 0.0
                            ),
                            **visual_context,
                        )
                    )
                loss, diagnostics = clipped_grpo_loss(
                    current_log_probs=current_log_probs,
                    old_log_probs=candidate.old_log_probs,
                    reference_log_probs=candidate.reference_log_probs,
                    advantage=candidate.advantage,
                    clip_eps=self.config.clip_eps,
                    kl_coef=self.config.kl_coef,
                )
                if self.config.preserve_during_grpo:
                    source_frames, source_context = self._source_visual_context(sample)
                    _, source_state = self.model.response_token_log_probs_with_state(
                        source_frames,
                        prompt,
                        candidate.response,
                        require_grad=False,
                        require_diagnostic_slots=False,
                        **source_context,
                    )
                    presence, severity = encode_degradation_profile(
                        [sample["degradation_profile"]],
                        self.model.degradation_factor_names,
                        device=self.model.device,
                    )
                    mask = rasterize_logged_occlusions(
                        sample, degraded_state.pathway_output, self.model.device
                    )
                    timestamps = torch.tensor(
                        [sample["anchor_timestamps_sec"]], device=self.model.device
                    )
                    deg, rel, cons = self.model.auxiliary_control_losses(
                        degraded_state,
                        source_state,
                        factor_presence=presence,
                        factor_severity=severity,
                        occlusion_token_mask=mask,
                        timestamps=timestamps,
                        compute_consistency=self.config.lambda_c_rl > 0.0,
                        motion_target_mode=self.config.reliability_target,
                        occlusion_mask_adjustment=(
                            self.config.occlusion_mask_adjustment
                        ),
                    )
                    loss = stage2_loss(
                        loss,
                        deg,
                        rel,
                        cons,
                        AuxiliaryLossWeights(
                            self.config.lambda_d_rl,
                            self.config.lambda_q_rl,
                            self.config.lambda_c_rl,
                        ),
                    ).total
                require_finite(loss, "GRPO loss")
                self.scaler.scale(loss / len(candidates)).backward()
                update_losses.append(loss.detach())
                diagnostic_rows.append(diagnostics)

            self.scaler.unscale_(self.optimizer)
            self.model.synchronize_visual_gradients()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for _, parameter in self.model.trainable_policy_named_parameters(
                        self.config.policy_scope
                    )
                ],
                self.config.max_grad_norm,
            )
            require_finite(gradient_norm, "GRPO gradient norm")
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if (
                self.model.motion_teacher is not None
                and self.config.policy_scope == "full"
                and self.config.ema_update_after_optimizer_step
            ):
                self.model.update_motion_teacher()
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

    @staticmethod
    def _source_visual_context(sample: dict):
        source = (sample["source_frames"].permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")
        motion = (
            sample["source_motion_frames"].permute(0, 2, 3, 1).cpu().numpy() * 255
        ).astype("uint8")
        return (
            [source[index] for index in range(source.shape[0])],
            {
                "motion_frames": [motion[index] for index in range(motion.shape[0])],
                "elapsed_seconds": sample["source_motion_elapsed_sec"],
                "timestamps": sample["anchor_timestamps_sec"],
            },
        )

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
                motion_tensor = sample["motion_frames"]
                motion_array = (
                    motion_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255
                ).astype("uint8")
                visual_context = (
                    {
                        "motion_frames": [
                            motion_array[index]
                            for index in range(motion_array.shape[0])
                        ],
                        "elapsed_seconds": sample["motion_elapsed_sec"],
                        "timestamps": sample["anchor_timestamps_sec"],
                    }
                    if self.model.reliability_pathway is not None
                    else {}
                )
                prompt = sample["prompt"]

                candidates = self._collect_rollout(
                    frames, prompt, sample, visual_context
                )
                optimization = self._optimize_rollout(
                    frames, prompt, candidates, visual_context, sample
                )
                self.rollout_step += 1
                components = {
                    key: sum(
                        getattr(candidate.rewards, key)
                        for candidate in candidates
                    )
                    / len(candidates)
                    for key in (
                        "rd",
                        "re",
                        "rt",
                        "rl",
                        "total",
                        "parse_success",
                        "event_active",
                        "temporal_active",
                        "active_fields_valid",
                    )
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
        if self.model.reliability_pathway is None:
            self.model.save_lora(path)
        else:
            self.model.save_core(path)
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
        torch.save(self.scaler.state_dict(), os.path.join(path, "scaler.pt"))
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
