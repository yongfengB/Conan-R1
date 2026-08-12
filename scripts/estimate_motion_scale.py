#!/usr/bin/env python3
"""Estimate and freeze v_max from native-rate flow on training sources only."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset.video_utils import estimate_training_velocity_scale, load_video


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/surv_vau")
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument("--output", default="data/surv_vau/motion_scale.json")
    args = parser.parse_args()
    root = Path(args.data_dir)
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
        training_sources.setdefault(source_id, (source_path, float(record["fps"])))
    if not training_sources:
        raise ValueError("No training sources were found.")
    videos = []
    fps_values = []
    for source_id in sorted(training_sources):
        source_path, fps = training_sources[source_id]
        videos.append(load_video(str(source_path)))
        fps_values.append(fps)
    value = estimate_training_velocity_scale(
        videos, fps_values, quantile=args.quantile
    )
    payload = {
        "schema_version": 1,
        "v_max": value,
        "unit": "pixels_per_second",
        "estimator": "opencv_farneback_reference",
        "quantile": args.quantile,
        "source_split": ["sft_train", "rl_train"],
        "source_count": len(training_sources),
        "annotations_sha256": sha256(annotations_path),
        "splits_sha256": sha256(splits_path),
        "split_manifest_sha256": sha256(split_manifest_path),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
