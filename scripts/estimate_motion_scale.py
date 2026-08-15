#!/usr/bin/env python3
"""Estimate v_max with the exact streamed 224x224 training motion path."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset.video_utils import (
    estimate_training_velocity_scale,
    motion_scale_contract,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/surv_vau")
    parser.add_argument("--method_config", default="configs/method_config.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--work_dir", default=None)
    args = parser.parse_args()
    root = Path(args.data_dir)
    method_path = Path(args.method_config)
    method = yaml.safe_load(method_path.read_text(encoding="utf-8"))
    normalization = method["motion"]["normalization"]
    contract = motion_scale_contract(method)
    preprocessing = contract["motion_preprocessing"]
    sampling = contract["sampling"]
    anchors = int(preprocessing["anchors"])
    frame_size = int(preprocessing["frame_size"][0])
    offset = int(preprocessing["native_frame_offset"])
    parameters = preprocessing["flow_parameters"]
    quantile = float(contract["quantile"])
    splits_path = root / "splits.json"
    annotations_path = root / "annotations.jsonl"
    split_manifest_path = root / "split_manifest.json"
    if not split_manifest_path.is_file():
        raise FileNotFoundError("split_manifest.json is required before estimation.")
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if split_manifest.get("annotations_sha256") != sha256(annotations_path):
        raise ValueError("split_manifest annotations hash does not match the dataset.")
    if split_manifest.get("splits_sha256") != sha256(splits_path):
        raise ValueError("split_manifest splits hash does not match the dataset.")
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in annotations_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    training_sources = {}
    for record in records:
        if splits[record["video_id"]] not in {"sft_train", "rl_train"}:
            continue
        source_id = record["source_video_id"]
        source_path = root / record["source_video_file"]
        candidate = (source_path, float(record["fps"]))
        previous = training_sources.setdefault(source_id, candidate)
        if previous != candidate:
            raise ValueError(
                f"{source_id}: inconsistent source path or fps across annotations."
            )
    if not training_sources:
        raise ValueError("No training sources were found.")
    sources = [
        (
            source_id,
            str(training_sources[source_id][0]),
            training_sources[source_id][1],
        )
        for source_id in sorted(training_sources)
    ]
    estimate = estimate_training_velocity_scale(
        sources,
        n=anchors,
        size=(frame_size, frame_size),
        offset=offset,
        quantile=quantile,
        parameters=parameters,
        samples_per_source=int(sampling["samples_per_source"]),
        sampling_seed=int(sampling["seed"]),
        work_dir=args.work_dir,
    )
    payload = {
        "schema_version": 2,
        "v_max": estimate.v_max,
        "unit": "pixels_per_second",
        "estimator": contract["estimator"],
        "quantile": quantile,
        "motion_preprocessing": preprocessing,
        "sampling": {
            "method": sampling["method"],
            "seed": estimate.sampling_seed,
            "samples_per_source": estimate.samples_per_source,
            "sampled_values": estimate.sampled_values,
            "storage": sampling["storage"],
        },
        "source_split": ["sft_train", "rl_train"],
        "source_count": estimate.source_count,
        "annotations_sha256": sha256(annotations_path),
        "splits_sha256": sha256(splits_path),
        "split_manifest_sha256": sha256(split_manifest_path),
        "method_config_sha256": sha256(method_path),
    }
    output = Path(args.output or normalization["v_max_file"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
