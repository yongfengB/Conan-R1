#!/usr/bin/env python3
"""Create immutable GRPO ablation YAML files from the experiment matrix."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import yaml


def deep_set(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cursor = config
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def materialize(
    matrix_path: Path, output_dir: Path, overwrite: bool = False
) -> Dict[str, str]:
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    block = matrix["grpo_ablations"]
    base_path = Path(block["base_config"])
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}

    variants = list(block["variants"])
    structural = matrix["structural_ablation"]
    variants.append(
        {
            "name": structural["name"],
            "overrides": structural["overrides"],
        }
    )
    for variant in variants:
        resolved = copy.deepcopy(base)
        for dotted_key, value in variant["overrides"].items():
            deep_set(resolved, dotted_key, value)
        resolved.setdefault("output", {})["checkpoint_dir"] = (
            f"checkpoints/grpo_{variant['name']}"
        )
        target = output_dir / f"{variant['name']}.yaml"
        if target.exists() and not overwrite:
            raise FileExistsError(
                f"{target} already exists; pass --overwrite to regenerate it."
            )
        payload = yaml.safe_dump(resolved, sort_keys=False)
        target.write_text(payload, encoding="utf-8")
        hashes[str(target)] = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    manifest = output_dir / "SHA256SUMS.json"
    manifest.write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix", default="experiments/experiment_matrix.yaml"
    )
    parser.add_argument("--output_dir", default="configs/generated")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    hashes = materialize(
        Path(args.matrix), Path(args.output_dir), overwrite=args.overwrite
    )
    print(json.dumps(hashes, indent=2))


if __name__ == "__main__":
    main()
