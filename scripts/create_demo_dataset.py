#!/usr/bin/env python3
"""Create the small, redistributable Conan-R1 audit dataset.

The demo is synthetic and is not a source of manuscript-scale results.  It is
large enough to exercise source-level splitting, all four degradation domains,
the strict output parser, raw-prediction scoring, and the tIoU calculation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.create_data_splits import file_sha256, stratified_partition


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def record(
    source_id: str,
    suffix: str,
    domain: str,
    level: float,
    profile: List[List[object]],
    protocol: str,
    combination: str,
) -> dict:
    video_id = f"{source_id}__{suffix}"
    start = 1.000
    end = 2.750
    type_annotation = (
        "; ".join(f"{name}:{float(severity):.1f}" for name, severity in profile)
        if profile
        else "none"
    )
    answer = f"event_type: rear-end collision; interval: [{start:.3f}, {end:.3f}]"
    return {
        "schema_version": "surv-vau-annotation-v1",
        "video_id": video_id,
        "source_video_id": source_id,
        "source_dataset": "synthetic_demo",
        "prompt": "Identify the traffic event and its temporal interval.",
        "degradation_profile": profile,
        "degradation_level": level,
        "degradation_domain": domain,
        "degradation_combination": combination,
        "synthesis_applied": domain.startswith("synthetic_"),
        "degradation_protocol": protocol,
        "gt_interval": [start, end],
        "event_type": "rear-end collision",
        "event_aliases": ["rear end crash"],
        "reasoning_target_length": int(round(32 + 96 * level)),
        "reasoning_target_source": "deterministic_policy",
        "duration_sec": 4.0,
        "fps": 6.0,
        "num_source_frames": 25,
        "type_annotation": type_annotation,
        "influence_annotation": "The observation condition changes cue reliability.",
        "reasoning_annotation": "The following vehicle closes the gap before contact.",
        "conclusion_annotation": "A rear-end collision occurs.",
        "answer_annotation": answer,
        "answer_references": [answer],
    }


def base_frames(source_index: int, natural_source: bool = False) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    for frame_index in range(25):
        background = 18 if natural_source else 52
        frame = np.full((64, 64, 3), background, dtype=np.uint8)
        cv2.line(frame, (0, 48), (63, 48), (180, 180, 180), 2)
        lead_x = 36
        follow_x = min(31, 5 + frame_index)
        color_shift = source_index % 40
        cv2.rectangle(frame, (lead_x, 34), (lead_x + 14, 45), (40, 80 + color_shift, 220), -1)
        cv2.rectangle(frame, (follow_x, 36), (follow_x + 13, 47), (220, 100, 40), -1)
        frames.append(frame)
    return frames


def transformed_frames(frames: Iterable[np.ndarray], domain: str, level: float) -> List[np.ndarray]:
    result = [frame.copy() for frame in frames]
    if domain == "synthetic_seen":
        kernel = {0.2: 3, 0.4: 7, 0.8: 13}[level]
        result = [cv2.GaussianBlur(frame, (kernel, 1), 0) for frame in result]
    elif domain == "synthetic_unseen":
        compressed = []
        for frame in result:
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 20])
            if not ok:
                raise RuntimeError("Could not encode a demo JPEG frame")
            compressed.append(cv2.imdecode(encoded, cv2.IMREAD_COLOR))
        result = compressed
    return result


def write_video(path: Path, frames: Iterable[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 6.0, (64, 64)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create demo video: {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def structured_prediction(row: dict) -> str:
    return (
        f"<TYPE>{row['type_annotation']}<TYPE_END>"
        f"<INFLUENCE>{row['influence_annotation']}<INFLUENCE_END>"
        f"<REASONING>{row['reasoning_annotation']}<REASONING_END>"
        f"<CONCLUSION>{row['conclusion_annotation']}<CONCLUSION_END>"
        f"<ANSWER>{row['answer_annotation']}<ANSWER_END>"
    )


def build(output: Path, seed: int) -> Dict[str, object]:
    base_records = [
        record(
            f"demo_source_{index:03d}",
            "clean",
            "clean",
            0.0,
            [],
            "source_observation",
            "none",
        )
        for index in range(20)
    ]
    by_source = {row["source_video_id"]: [row] for row in base_records}
    outer = stratified_partition(
        by_source,
        fractions=(0.70, 0.15, 0.15),
        names=("train", "val", "test"),
        seed=seed,
    )
    training = {
        source_id: by_source[source_id]
        for source_id, split in outer.items()
        if split == "train"
    }
    inner = stratified_partition(
        training,
        fractions=(0.30, 0.70),
        names=("sft_train", "rl_train"),
        seed=seed + 1,
    )
    source_split = {
        source_id: inner[source_id] if split == "train" else split
        for source_id, split in outer.items()
    }

    rows = list(base_records)
    for source_id in sorted(source for source, split in source_split.items() if split == "test"):
        rows.extend(
            [
                record(source_id, "seen20", "synthetic_seen", 0.2, [["motion_blur", 0.2]], "surv-vau-degradation-v1", "single_operator"),
                record(source_id, "seen40", "synthetic_seen", 0.4, [["motion_blur", 0.4]], "surv-vau-degradation-v1", "single_operator"),
                record(source_id, "seen80", "synthetic_seen", 0.8, [["motion_blur", 0.8]], "surv-vau-degradation-v1", "single_operator"),
                record(source_id, "unseen80", "synthetic_unseen", 0.8, [["compression_artifact", 0.8]], "surv-vau-degradation-v1", "held_out_operator"),
                record(source_id, "natural", "natural", 0.0, [], "source_observation", "none"),
            ]
        )
    rows.sort(key=lambda item: item["video_id"])

    output.mkdir(parents=True, exist_ok=True)
    annotations = output / "annotations.jsonl"
    annotations.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    split_map = {
        row["video_id"]: source_split[row["source_video_id"]] for row in rows
    }
    splits = output / "splits.json"
    splits.write_bytes(canonical_json(split_map))

    for row in rows:
        source_index = int(row["source_video_id"].rsplit("_", 1)[1])
        frames = base_frames(source_index, natural_source=row["degradation_domain"] == "natural")
        frames = transformed_frames(frames, row["degradation_domain"], float(row["degradation_level"]))
        write_video(output / "videos" / f"{row['video_id']}.mp4", frames)

    predictions_path = REPO_ROOT / "results" / "demo_raw_predictions.jsonl"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    test_rows = [row for row in rows if split_map[row["video_id"]] == "test"]
    predictions_path.write_text(
        "".join(
            json.dumps(
                {"video_id": row["video_id"], "raw_output": structured_prediction(row)},
                sort_keys=True,
            )
            + "\n"
            for row in test_rows
        ),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "dataset_version": "conan-r1-synthetic-demo-v1",
        "artifact_role": "executable_schema_parser_metric_demo_not_paper_evidence",
        "split_rule_id": "source-stratified-70-15-15_then-train-30-70-v1",
        "seed": seed,
        "outer_fractions": {"train": 0.70, "val": 0.15, "test": 0.15},
        "inner_train_fractions": {"sft_train": 0.30, "rl_train": 0.70},
        "stratification": ["source_dataset", "event_type"],
        "source_video_isolation": True,
        "source_count": len(source_split),
        "instance_count": len(rows),
        "source_counts": {
            name: sum(value == name for value in source_split.values())
            for name in ("sft_train", "rl_train", "val", "test")
        },
        "instance_counts": {
            name: sum(value == name for value in split_map.values())
            for name in ("sft_train", "rl_train", "val", "test")
        },
        "annotations_sha256": file_sha256(annotations),
        "splits_sha256": file_sha256(splits),
        "raw_predictions_sha256": file_sha256(predictions_path),
    }
    (output / "split_manifest.json").write_bytes(canonical_json(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data" / "demo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = build(args.output, args.seed)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
