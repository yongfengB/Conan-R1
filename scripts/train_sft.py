#!/usr/bin/env python3
"""Train the Conan-R1 SFT adapter from the paper configuration."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset.dataset import SurvVAUDataset
from model.conan_r1 import ConanR1Model, LoRAConfig
from training.sft_trainer import SFTConfig, SFTTrainer
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
    parser = argparse.ArgumentParser(description="Conan-R1 supervised fine-tuning")
    parser.add_argument("--config", default="configs/sft_config.yaml")
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:1, or cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    raw = load_config(args.config)
    train_cfg = SFTConfig.from_yaml(args.config)
    data_cfg = raw["data"]
    model_cfg = raw["model"]

    require_dataset(data_cfg["data_dir"])
    device, rank, world_size = init_distributed(args.device)
    seed_everything(train_cfg.seed + rank)
    requested_splits = data_cfg.get(
        "splits", [data_cfg.get("split", "sft_train")]
    )
    dataset = SurvVAUDataset(
        data_dir=data_cfg["data_dir"],
        split=requested_splits,
        num_frames=int(data_cfg.get("num_frames", 25)),
        frame_size=int(data_cfg.get("frame_size", 224)),
    )
    if not dataset:
        raise RuntimeError("The configured SFT split contains no samples.")

    lora = LoRAConfig(
        rank=int(model_cfg.get("lora_rank", 16)),
        alpha=int(model_cfg.get("lora_alpha", 32)),
        dropout=float(model_cfg.get("lora_dropout", 0.05)),
    )
    model = ConanR1Model(
        base_model=model_cfg["base_model"],
        lora_config=lora,
        device=device,
    )
    initial_checkpoint = model_cfg.get("init_checkpoint")
    if initial_checkpoint:
        if not Path(initial_checkpoint).exists():
            raise FileNotFoundError(
                f"Initial SFT checkpoint not found: {initial_checkpoint}"
            )
        model.load_lora(initial_checkpoint, is_trainable=True)
    if world_size > 1:
        model.enable_distributed(int(device.rsplit(":", 1)[1]))
    trainer = SFTTrainer(model, dataset, train_cfg)
    trainer.train()
    if is_main_process():
        write_run_metadata(
            train_cfg.checkpoint_dir,
            args.config,
            {
                "stage": "sft",
                "initial_checkpoint": initial_checkpoint,
                "splits": requested_splits,
                "num_training_samples": len(dataset),
                "seed": train_cfg.seed,
                "world_size": world_size,
            },
        )
    finish_distributed()


if __name__ == "__main__":
    main()
