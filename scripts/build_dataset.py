#!/usr/bin/env python3
"""Build a deterministic degradation-controlled Surv-VAU-style dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset.builder import SurvVAUBuilder
from dataset.video_utils import load_video, native_motion_pairs
from model.conan_r1 import ConanR1Model
from scripts._common import resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build structured Surv-VAU data")
    parser.add_argument("--source_dir", action="append", required=True)
    parser.add_argument(
        "--annotation_file",
        required=True,
        help="Source annotations including the frozen degradation_profiles list",
    )
    parser.add_argument("--output_dir", default="data/surv_vau")
    parser.add_argument(
        "--annotator_model", default="Qwen/Qwen2.5-VL-3B-Instruct"
    )
    parser.add_argument(
        "--annotator_revision",
        default="c747f21f03e7d0792c30766310bd7d8de17eeeb3",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _write_video(path: Path, frames, fps: float) -> None:
    if not frames:
        raise ValueError(f"Cannot write an empty clip: {path}")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output = Path(args.output_dir)
    annotations_path = output / "annotations.jsonl"
    splits_path = output / "splits.json"
    if annotations_path.exists() or splits_path.exists():
        raise FileExistsError(
            f"{output} already contains a dataset manifest; choose a new output directory."
        )
    videos_dir = output / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    annotator = ConanR1Model(
        base_model=args.annotator_model,
        base_model_revision=args.annotator_revision,
        device=resolve_device(args.device),
        enable_lora=False,
    )
    splits = SurvVAUBuilder(annotator, seed=args.seed).build(
        args.source_dir, args.annotation_file
    )

    split_map = {}
    source_files = {}
    for source_dir in args.source_dir:
        for source_path in Path(source_dir).glob("*.mp4"):
            source_files.setdefault(source_path.stem, source_path)
    with open(annotations_path, "w", encoding="utf-8") as annotations:
        for split_name, samples in splits.items():
            for sample in samples:
                pairs = native_motion_pairs(sample.num_source_frames, n=25, offset=1)
                source_name = f"{sample.source_video_id}__source.mp4"
                source_output = videos_dir / source_name
                if not source_output.exists():
                    if sample.source_video_id not in source_files:
                        raise FileNotFoundError(
                            f"Source pair not found for {sample.source_video_id}"
                        )
                    _write_video(
                        source_output,
                        load_video(str(source_files[sample.source_video_id])),
                        sample.fps,
                    )
                record = {
                    "schema_version": "surv-vau-annotation-v2",
                    "video_id": sample.video_id,
                    "source_video_id": sample.source_video_id,
                    "source_dataset": sample.source_dataset,
                    "prompt": sample.prompt,
                    "degradation_profile": sample.degradation_profile,
                    "degradation_level": sample.difficulty_level,
                    "degradation_domain": (
                        sample.degradation_domain
                    ),
                    "degradation_combination": "+".join(
                        factor for factor, _ in sample.degradation_profile
                    ) or "none",
                    "synthesis_applied": sample.degradation_domain.startswith(
                        "synthetic_"
                    ),
                    "degradation_protocol": sample.synthesis_metadata.get(
                        "protocol", "surv-vau-degradation-v1"
                    ),
                    "synthesis_metadata": sample.synthesis_metadata,
                    "gt_interval": list(sample.gt_interval),
                    "event_type": sample.event_type,
                    "event_aliases": sample.event_aliases,
                    "task_mask": sample.task_mask,
                    "source_video_file": f"videos/{source_name}",
                    "anchor_indices": [first for first, _ in pairs],
                    "motion_pair_indices": [list(pair) for pair in pairs],
                    "motion_elapsed_sec": [
                        (second - first) / sample.fps for first, second in pairs
                    ],
                    "influence_targets": sample.influence_targets,
                    "occlusion_token_mask": None,
                    "reasoning_target_length": sample.reasoning_target_length,
                    "reasoning_target_source": sample.reasoning_target_source,
                    "duration_sec": sample.duration_sec,
                    "fps": sample.fps,
                    "num_source_frames": sample.num_source_frames,
                    "type_annotation": sample.type_annotation,
                    "influence_annotation": sample.influence_annotation,
                    "reasoning_annotation": sample.reasoning_annotation,
                    "conclusion_annotation": sample.conclusion_annotation,
                    "answer_annotation": sample.answer_annotation,
                    "split": split_name,
                }
                annotations.write(json.dumps(record, ensure_ascii=False) + "\n")
                split_map[sample.video_id] = split_name
                _write_video(
                    videos_dir / f"{sample.video_id}.mp4", sample.frames, sample.fps
                )

    with open(splits_path, "w", encoding="utf-8") as handle:
        json.dump(split_map, handle, indent=2)
    split_manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "annotations_sha256": _sha256(annotations_path),
        "splits_sha256": _sha256(splits_path),
        "instance_counts": dict(Counter(split_map.values())),
        "source_video_isolation": True,
        "builder": "SurvVAUBuilder",
    }
    (output / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(split_map)} instances to {output}")


if __name__ == "__main__":
    main()
