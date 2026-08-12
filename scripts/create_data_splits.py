#!/usr/bin/env python3
"""Create deterministic source-video-level 70/15/15 and 30/70 splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset.splitting import stratified_partition


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/surv_vau")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.data_dir)
    annotations_path = root / "annotations.jsonl"
    split_path = root / "splits.json"
    if split_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{split_path} already exists; pass --overwrite only after review."
        )

    records = [
        json.loads(line)
        for line in annotations_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_source = defaultdict(list)
    for record in records:
        by_source[record["source_video_id"]].append(record)

    outer = stratified_partition(
        by_source,
        fractions=(0.70, 0.15, 0.15),
        names=("train", "val", "test"),
        seed=args.seed,
    )
    training_sources = {
        source_id: by_source[source_id]
        for source_id, split in outer.items()
        if split == "train"
    }
    inner = stratified_partition(
        training_sources,
        fractions=(0.30, 0.70),
        names=("sft_train", "rl_train"),
        seed=args.seed + 1,
    )
    source_split = {
        source_id: (
            inner[source_id] if split == "train" else split
        )
        for source_id, split in outer.items()
    }
    invalid_held_out = [
        record["video_id"]
        for record in records
        if record.get("degradation_domain") in {"synthetic_unseen", "natural"}
        and source_split[record["source_video_id"]] != "test"
    ]
    if invalid_held_out:
        raise ValueError(
            "Held-out-domain variants exist for non-test sources; construct "
            "synthetic-unseen/natural variants only after the source split. "
            f"Invalid instances: {len(invalid_held_out)}."
        )
    instance_split = {
        record["video_id"]: source_split[record["source_video_id"]]
        for record in records
    }
    split_path.write_text(
        json.dumps(instance_split, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "split_rule_id": "source-stratified-70-15-15_then-train-30-70-v1",
        "seed": args.seed,
        "outer_fractions": {"train": 0.70, "val": 0.15, "test": 0.15},
        "inner_train_fractions": {"sft_train": 0.30, "rl_train": 0.70},
        "annotations_sha256": file_sha256(annotations_path),
        "splits_sha256": file_sha256(split_path),
        "source_counts": dict(Counter(source_split.values())),
        "instance_counts": dict(Counter(instance_split.values())),
        "stratification": ["source_dataset", "event_type"],
        "source_video_isolation": True,
    }
    (root / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
