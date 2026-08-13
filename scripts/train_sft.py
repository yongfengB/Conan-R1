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
from model.reliability_pathway import ReliabilityPathwayConfig
from training.sft_trainer import SFTConfig, SFTTrainer
from scripts._common import (
    finish_distributed,
    init_distributed,
    is_main_process,
    load_config,
    require_dataset,
    seed_everything,
    sha256_file,
    write_run_metadata,
)


def load_motion_vmax(raw: dict) -> float:
    method = load_config(raw["model"]["method_config"])
    path = Path(method["motion"]["normalization"]["v_max_file"])
    if not path.is_file():
        raise FileNotFoundError(
            f"Training-split motion scale not found: {path}. "
            "Create it from the training split; do not estimate it per video."
        )
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    data_root = Path(raw["data"]["data_dir"])
    expected = {
        "annotations_sha256": sha256_file(data_root / "annotations.jsonl"),
        "splits_sha256": sha256_file(data_root / "splits.json"),
        "split_manifest_sha256": sha256_file(data_root / "split_manifest.json"),
    }
    mismatches = {
        key: {"motion_scale": payload.get(key), "dataset": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "motion_scale.json is not bound to the configured dataset: "
            f"{mismatches}"
        )
    if payload.get("unit") != "pixels_per_second":
        raise ValueError("motion_scale.json must use pixels_per_second.")
    value = float(payload["v_max"])
    if not value > 0.0:
        raise ValueError("motion_scale.json v_max must be positive.")
    return value


def load_motion_flow_parameters(raw: dict) -> dict:
    from dataset.video_utils import validate_farneback_parameters

    method = load_config(raw["model"]["method_config"])
    return validate_farneback_parameters(method["motion"]["flow_parameters"])


def build_reliability_config(raw: dict, base_model: str) -> ReliabilityPathwayConfig:
    method_path = Path(raw["model"]["method_config"])
    method = load_config(str(method_path))
    from transformers import AutoConfig

    backbone = AutoConfig.from_pretrained(
        base_model,
        revision=raw["model"].get("base_model_revision"),
        trust_remote_code=True,
    )
    appearance_dim = int(backbone.vision_config.out_hidden_size)
    output_dim = int(backbone.hidden_size)
    path = method["reliability_pathway"]
    pathway_override = raw.get("model", {}).get("pathway", {})

    def setting(name: str):
        return pathway_override.get(name, path[name])

    return ReliabilityPathwayConfig(
        appearance_dim=appearance_dim,
        hidden_dim=int(path["hidden_dim"]),
        output_dim=output_dim,
        degradation_dim=int(path["degradation_dim"]),
        num_factors=len(method["degradation_factors"]),
        max_anchors=int(method["appearance_encoder"]["anchors"]),
        max_spatial_tokens=int(path["max_spatial_tokens"]),
        q_min=float(path["q_min"]),
        reliability_prior_scale=float(path["reliability_prior_scale"]),
        tau_appearance=float(path["tau_appearance"]),
        tau_motion=float(path["tau_motion"]),
        ema_decay=float(path["ema_decay"]),
        dropout=float(path["dropout"]),
        use_reliability_fusion=bool(setting("use_reliability_fusion")),
        use_event_aware_pooling=bool(setting("use_event_aware_pooling")),
        use_temporal_reliability=bool(setting("use_temporal_reliability")),
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
        enabled_blocks=train_cfg.enabled_blocks,
    )
    if not dataset:
        raise RuntimeError("The configured SFT split contains no samples.")

    lora = LoRAConfig(
        rank=int(model_cfg.get("lora_rank", 16)),
        alpha=int(model_cfg.get("lora_alpha", 32)),
        dropout=float(model_cfg.get("lora_dropout", 0.05)),
    )
    full_policy = train_cfg.policy_scope == "full"
    model = ConanR1Model(
        base_model=model_cfg["base_model"],
        base_model_revision=model_cfg.get("base_model_revision"),
        lora_config=lora,
        device=device,
        reliability_config=(
            build_reliability_config(raw, model_cfg["base_model"])
            if full_policy
            else None
        ),
        motion_v_max=load_motion_vmax(raw) if full_policy else None,
        degradation_factor_names=(
            load_config(model_cfg["method_config"])["degradation_factors"]
            if full_policy
            else None
        ),
        motion_flow_parameters=(
            load_motion_flow_parameters(raw) if full_policy else None
        ),
    )
    initial_checkpoint = model_cfg.get("init_checkpoint")
    if initial_checkpoint:
        if not Path(initial_checkpoint).exists():
            raise FileNotFoundError(
                f"Initial SFT checkpoint not found: {initial_checkpoint}"
            )
        if full_policy:
            model.load_core(initial_checkpoint, is_trainable=True)
        else:
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
