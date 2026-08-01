#!/usr/bin/env python3
"""Audit data exposure and optimizer-step budgets for the matched controls."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def selected_count(split_map: dict, configured_splits) -> int:
    requested = (
        {configured_splits}
        if isinstance(configured_splits, str)
        else set(configured_splits)
    )
    return sum(split in requested for split in split_map.values())


def sft_budget(path: Path, split_map: dict, world_size: int) -> dict:
    config = load_yaml(path)
    training = config["training"]
    splits = config["data"].get(
        "splits", [config["data"].get("split", "sft_train")]
    )
    count = selected_count(split_map, splits)
    local_samples = math.ceil(count / world_size)
    steps_per_epoch = math.ceil(
        local_samples
        / (
            int(training.get("batch_size", 1))
            * int(training["gradient_accumulation_steps"])
        )
    )
    epochs = int(training["epochs"])
    return {
        "stage": "sft",
        "config": str(path),
        "splits": list(splits),
        "instances_per_epoch": count,
        "data_epochs": epochs,
        "optimizer_steps_per_rank": steps_per_epoch * epochs,
        "generated_candidates_per_rank": 0,
    }


def grpo_budget(path: Path, split_map: dict, world_size: int) -> dict:
    config = load_yaml(path)
    training = config["training"]
    split = config["data"].get("split", "rl_train")
    count = selected_count(split_map, split)
    local_samples = math.ceil(count / world_size)
    epochs = int(training["epochs"])
    update_epochs = int(training["update_epochs"])
    return {
        "stage": "grpo",
        "config": str(path),
        "splits": [split],
        "instances_per_epoch": count,
        "data_epochs": epochs,
        "optimizer_steps_per_rank": local_samples * epochs * update_epochs,
        "generated_candidates_per_rank": (
            local_samples * epochs * int(training["group_size"])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/surv_vau")
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument(
        "--output", default="results/training_budget_audit.json"
    )
    args = parser.parse_args()
    if args.world_size < 1:
        raise ValueError("world_size must be positive.")

    split_path = Path(args.data_dir) / "splits.json"
    split_map = json.loads(split_path.read_text(encoding="utf-8"))
    rows = [
        sft_budget(Path("configs/sft_config.yaml"), split_map, args.world_size),
        sft_budget(
            Path("configs/continued_sft_config.yaml"),
            split_map,
            args.world_size,
        ),
        sft_budget(
            Path("configs/continued_sft_update_matched_config.yaml"),
            split_map,
            args.world_size,
        ),
        sft_budget(
            Path("configs/sft100_config.yaml"), split_map, args.world_size
        ),
        grpo_budget(
            Path("configs/grpo_config.yaml"), split_map, args.world_size
        ),
    ]
    payload = {
        "world_size": args.world_size,
        "split_instance_counts": dict(Counter(split_map.values())),
        "rows": rows,
        "interpretation": {
            "continued_sft70": "data-epoch-matched control",
            "continued_sft70_update_matched": "optimizer-step-matched control",
            "warning": (
                "Optimizer-step matching is not FLOP matching: GRPO samples a "
                "candidate group and performs reference-policy forwards."
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
