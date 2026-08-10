#!/usr/bin/env python3
"""Evaluate evidence deletion and structured-rationale interventions."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset.dataset import SurvVAUDataset
from dataset.video_utils import uniform_sample_timestamps
from evaluation.evaluator import Evaluator
from model.conan_r1 import ConanR1Model
from model.parser import parse_structured_output
from scripts._common import (
    collect_runtime_metadata,
    require_dataset,
    resolve_device,
    seed_everything,
)


WRONG_FACTORS = (
    "local_occlusion",
    "motion_blur",
    "lens_flare",
    "low_light",
    "rain_snow",
    "fog",
)


def frame_list(sample: dict):
    array = (
        sample["frames"].permute(0, 2, 3, 1).numpy() * 255
    ).astype("uint8")
    return [array[index] for index in range(array.shape[0])]


def mask_event_frames(frames, sample: dict):
    result = [frame.copy() for frame in frames]
    timestamps = uniform_sample_timestamps(
        sample["num_source_frames"], sample["fps"], len(result)
    )
    start, end = map(float, sample["gt_interval"])
    selected = [
        index
        for index, timestamp in enumerate(timestamps)
        if start <= timestamp <= end
    ]
    if not selected:
        midpoint = (start + end) / 2.0
        selected = [int(np.argmin(np.abs(timestamps - midpoint)))]
    for index in selected:
        result[index][:] = 0
    return result


def deterministic_shuffle(frames, video_id: str, seed: int):
    digest = hashlib.sha256(f"{seed}:{video_id}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    order = rng.permutation(len(frames))
    return [frames[int(index)] for index in order]


def wrong_factor(sample: dict) -> str:
    active = {str(item[0]).lower() for item in sample["degradation_profile"]}
    for factor in WRONG_FACTORS:
        if factor not in active:
            return factor
    return "sensor_noise"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default="data/surv_vau")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--num_frames", type=int, default=25)
    parser.add_argument("--frame_size", type=int, default=224)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output", default="results/intervention_evaluation.json"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    require_dataset(args.data_dir)
    seed_everything(args.seed)
    dataset = SurvVAUDataset(
        args.data_dir,
        args.split,
        num_frames=args.num_frames,
        frame_size=args.frame_size,
    )
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model = ConanR1Model(
        args.base_model, device=resolve_device(args.device), enable_lora=True
    )
    model.load_lora(args.checkpoint, is_trainable=False)

    conditions = defaultdict(list)
    references = []
    limit = len(dataset) if args.max_samples is None else min(
        len(dataset), args.max_samples
    )
    for index in range(limit):
        sample = dataset[index]
        frames = frame_list(sample)
        original = model.generate(
            frames,
            sample["prompt"],
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        conditions["original"].append(original)
        conditions["event_evidence_masked"].append(
            model.generate(
                mask_event_frames(frames, sample),
                sample["prompt"],
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        )
        conditions["temporal_order_shuffled"].append(
            model.generate(
                deterministic_shuffle(frames, sample["video_id"], args.seed),
                sample["prompt"],
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        )

        parsed = parse_structured_output(original)
        if parsed is None:
            conditions["type_permuted"].append("")
            conditions["reasoning_removed"].append("")
            conditions["rationale_only_no_video"].append("")
        else:
            type_prefix = (
                f"<TYPE>{wrong_factor(sample)}:0.8<TYPE_END>"
                f"<INFLUENCE>{parsed.influence_block}<INFLUENCE_END>"
                "<REASONING>"
            )
            conditions["type_permuted"].append(
                model.generate_with_prefix(
                    frames,
                    sample["prompt"],
                    type_prefix,
                    max_new_tokens=args.max_new_tokens,
                )
            )
            removed_prefix = (
                f"<TYPE>{parsed.type_block}<TYPE_END>"
                f"<INFLUENCE>{parsed.influence_block}<INFLUENCE_END>"
                "<REASONING><REASONING_END><CONCLUSION>"
            )
            conditions["reasoning_removed"].append(
                model.generate_with_prefix(
                    frames,
                    sample["prompt"],
                    removed_prefix,
                    max_new_tokens=args.max_new_tokens,
                )
            )
            rationale_prefix = (
                f"<TYPE>{parsed.type_block}<TYPE_END>"
                f"<INFLUENCE>{parsed.influence_block}<INFLUENCE_END>"
                f"<REASONING>{parsed.reasoning_block}<REASONING_END>"
                "<CONCLUSION>"
            )
            blank_frames = [np.zeros_like(frame) for frame in frames]
            conditions["rationale_only_no_video"].append(
                model.generate_with_prefix(
                    blank_frames,
                    sample["prompt"],
                    rationale_prefix,
                    max_new_tokens=args.max_new_tokens,
                )
            )
        references.append(sample)

    evaluations = {}
    for condition, outputs in conditions.items():
        metrics, per_sample = Evaluator().evaluate(outputs, references)
        evaluations[condition] = {
            "metrics": metrics,
            "per_sample": [
                {**row, "raw_output": output}
                for row, output in zip(per_sample, outputs)
            ],
        }

    original_metrics = evaluations["original"]["metrics"]
    deltas = {
        condition: {
            metric: float(result["metrics"].get(metric, 0.0))
            - float(original_metrics.get(metric, 0.0))
            for metric in (
                "Event-Accuracy",
                "Event-Macro-F1",
                "tIoU",
                "METEOR",
                "ROUGE-L",
                "Parse-Success",
            )
        }
        for condition, result in evaluations.items()
        if condition != "original"
    }
    payload = {
        "protocol": {
            "checkpoint": args.checkpoint,
            "split": args.split,
            "seed": args.seed,
            "num_samples": limit,
            "unit": "single fixed run",
            "interpretation": (
                "These are behavioral sensitivity tests. They do not establish "
                "formal causal identification."
            ),
        },
        "evaluations": evaluations,
        "delta_from_original": deltas,
        "provenance": collect_runtime_metadata(
            data_dir=args.data_dir, checkpoint=args.checkpoint
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(deltas, indent=2))


if __name__ == "__main__":
    main()
