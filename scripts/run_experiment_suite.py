#!/usr/bin/env python3
"""Print or execute the fixed single-run Conan-R1 experiment suite.

Execution is opt-in via ``--execute``.  The default mode is a safe dry run that
also checks whether required data and preceding checkpoints exist.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

import yaml

from materialize_experiments import materialize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix", default="experiments/experiment_matrix.yaml"
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--include_ablations", action="store_true", default=False
    )
    args = parser.parse_args()

    matrix_path = Path(args.matrix)
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    data_root = Path("data/surv_vau")
    required_data = [
        data_root / "annotations.jsonl",
        data_root / "splits.json",
        data_root / "split_manifest.json",
        data_root / "videos",
    ]
    missing = [str(path) for path in required_data if not path.exists()]

    commands = [item["command"] for item in matrix["core_training"]]
    if args.include_ablations:
        materialize(
            matrix_path,
            Path("configs/generated"),
            overwrite=True,
        )
        commands.append(
            "torchrun --standalone --nproc_per_node=4 "
            "scripts/train_sft.py --config "
            + matrix["structural_ablation"]["sft_config"]
        )
        for variant in matrix["grpo_ablations"]["variants"]:
            commands.append(
                "torchrun --standalone --nproc_per_node=4 "
                f"scripts/train_grpo.py --config configs/generated/{variant['name']}.yaml"
            )
        for variant in matrix.get("reward_weight_sensitivity", {}).get(
            "variants", []
        ):
            commands.append(
                "torchrun --standalone --nproc_per_node=4 "
                f"scripts/train_grpo.py --config configs/generated/{variant['name']}.yaml"
            )
        commands.append(
            "torchrun --standalone --nproc_per_node=4 "
            "scripts/train_grpo.py --config "
            "configs/generated/without_type_influence.yaml"
        )

    report = {
        "single_run_protocol": True,
        "seed": matrix["seed"],
        "data_ready": not missing,
        "missing_inputs": missing,
        "commands": commands,
        "mode": "execute" if args.execute else "dry-run",
    }
    print(json.dumps(report, indent=2))
    if not args.execute:
        return
    if missing:
        raise FileNotFoundError(
            "Cannot execute the suite because required data are missing: "
            + ", ".join(missing)
        )
    for command in commands:
        print(f"\n[RUN] {command}", flush=True)
        subprocess.run(shlex.split(command), check=True)


if __name__ == "__main__":
    main()
