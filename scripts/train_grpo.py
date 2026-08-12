#!/usr/bin/env python3
"""Run Conan-R1 degradation-aware GRPO alignment."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset.dataset import SurvVAUDataset
from model.conan_r1 import ConanR1Model, LoRAConfig
from training.grpo_trainer import GRPOConfig, GRPOTrainer
from scripts.train_sft import build_reliability_config, load_motion_vmax
from scripts._common import (
    finish_distributed,
    init_distributed,
    is_main_process,
    load_config,
    require_dataset,
    seed_everything,
    write_run_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conan-R1 GRPO training")
    parser.add_argument("--config", default="configs/grpo_config.yaml")
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:1, or cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    raw = load_config(args.config)
    train_cfg = GRPOConfig.from_yaml(args.config)
    data_cfg = raw["data"]
    model_cfg = raw["model"]
    checkpoint = model_cfg["sft_checkpoint"]
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"SFT checkpoint not found: {checkpoint}")

    require_dataset(data_cfg["data_dir"])
    device, rank, world_size = init_distributed(args.device)
    seed_everything(train_cfg.seed + rank)
    dataset = SurvVAUDataset(
        data_dir=data_cfg["data_dir"],
        split=data_cfg.get("split", "rl_train"),
        num_frames=int(data_cfg.get("num_frames", 25)),
        frame_size=int(data_cfg.get("frame_size", 224)),
    )
    if not dataset:
        raise RuntimeError("The configured RL split contains no samples.")

    lora = LoRAConfig(
        rank=int(model_cfg.get("lora_rank", 16)),
        alpha=int(model_cfg.get("lora_alpha", 32)),
        dropout=float(model_cfg.get("lora_dropout", 0.05)),
    )
    reliability_config = build_reliability_config(raw, model_cfg["base_model"])
    motion_v_max = load_motion_vmax(raw)
    factor_names = load_config(model_cfg["method_config"])["degradation_factors"]
    model = ConanR1Model(
        model_cfg["base_model"],
        base_model_revision=model_cfg.get("base_model_revision"),
        lora_config=lora,
        device=device,
        reliability_config=reliability_config,
        motion_v_max=motion_v_max,
        degradation_factor_names=factor_names,
    )
    model.load_core(checkpoint, is_trainable=True)
    ref_model = ConanR1Model(
        model_cfg["base_model"],
        base_model_revision=model_cfg.get("base_model_revision"),
        lora_config=lora,
        device=device,
        reliability_config=reliability_config,
        motion_v_max=motion_v_max,
        degradation_factor_names=factor_names,
    )
    ref_model.load_core(checkpoint, is_trainable=False)
    if world_size > 1:
        model.enable_distributed(int(device.rsplit(":", 1)[1]))

    trainer = GRPOTrainer(model, ref_model, dataset, train_cfg)
    trainer.train()
    if is_main_process():
        write_run_metadata(
            train_cfg.checkpoint_dir,
            args.config,
            {
                "stage": "grpo",
                "num_training_samples": len(dataset),
                "seed": train_cfg.seed,
                "world_size": world_size,
            },
        )
    finish_distributed()


if __name__ == "__main__":
    main()
