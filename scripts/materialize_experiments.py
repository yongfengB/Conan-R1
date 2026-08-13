#!/usr/bin/env python3
"""Create immutable, complete YAML files for every manuscript control."""
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
    block_names = matrix.get(
        "materialized_blocks",
        ["stage1_architecture", "stage2_reward", "matched_stage2", "appendix_controls"],
    )
    blocks = [(name, matrix[name]) for name in block_names]
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {
        f"{variant['name']}.yaml"
        for _, block in blocks
        for variant in block["variants"]
        if variant.get("materialize", True)
    }
    if overwrite:
        for stale in output_dir.glob("*.yaml"):
            if stale.name not in expected_names:
                stale.unlink()
    hashes = {}

    seen = set()
    for block_name, block in blocks:
        for variant in block["variants"]:
            if not variant.get("materialize", True):
                continue
            name = variant["name"]
            if name in seen:
                raise ValueError(f"Duplicate materialized experiment name: {name}")
            seen.add(name)
            base_path = Path(variant.get("base_config", block["base_config"]))
            resolved = yaml.safe_load(base_path.read_text(encoding="utf-8"))
            resolved = copy.deepcopy(resolved)
            for dotted_key, value in variant.get("overrides", {}).items():
                deep_set(resolved, dotted_key, value)
            resolved.setdefault("experiment", {}).update(
                {
                    "matrix_block": block_name,
                    "variant": name,
                    "base_config": str(base_path),
                }
            )
            resolved.setdefault("output", {})["checkpoint_dir"] = variant.get(
                "checkpoint_dir", f"checkpoints/{name}"
            )
            target = output_dir / f"{name}.yaml"
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
