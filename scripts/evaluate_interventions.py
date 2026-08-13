#!/usr/bin/env python3
"""Fixed-checkpoint interventions on the predicted reliability field only."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset.dataset import SurvVAUDataset
from evaluation.evaluator import Evaluator
from model.conan_r1 import ConanR1Model, LoRAConfig
from model.reliability_pathway import (
    RELIABILITY_INTERVENTIONS,
    ReliabilityPathwayConfig,
)
from scripts._common import (
    collect_runtime_metadata,
    load_core_protocol,
    require_dataset,
    resolve_device,
    seed_everything,
)


def frame_lists(sample: dict):
    frames = (sample["frames"].permute(0, 2, 3, 1).numpy() * 255).astype("uint8")
    motion = (
        sample["motion_frames"].permute(0, 2, 3, 1).numpy() * 255
    ).astype("uint8")
    return (
        [frames[index] for index in range(frames.shape[0])],
        [motion[index] for index in range(motion.shape[0])],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default="data/surv_vau")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument(
        "--base_model_revision",
        default="c747f21f03e7d0792c30766310bd7d8de17eeeb3",
    )
    parser.add_argument("--num_frames", type=int, default=25)
    parser.add_argument("--frame_size", type=int, default=224)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="results/reliability_interventions.json")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    require_dataset(args.data_dir)
    seed_everything(args.seed)
    dataset = SurvVAUDataset(
        args.data_dir, args.split, args.num_frames, args.frame_size
    )
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    core_protocol = load_core_protocol(checkpoint)
    model = ConanR1Model(
        args.base_model,
        base_model_revision=args.base_model_revision,
        lora_config=LoRAConfig(),
        device=resolve_device(args.device),
        reliability_config=ReliabilityPathwayConfig(
            **core_protocol["reliability_config"]
        ),
        motion_v_max=float(core_protocol["motion_v_max"]),
        degradation_factor_names=core_protocol["degradation_factor_names"],
        motion_flow_parameters=core_protocol["motion_flow_parameters"],
    )
    model.load_core(args.checkpoint, is_trainable=False)

    conditions = defaultdict(list)
    references = []
    limit = len(dataset) if args.max_samples is None else min(
        len(dataset), args.max_samples
    )
    for index in range(limit):
        sample = dataset[index]
        frames, motion_frames = frame_lists(sample)
        context = {
            "motion_frames": motion_frames,
            "elapsed_seconds": sample["motion_elapsed_sec"],
            "timestamps": sample["anchor_timestamps_sec"],
            "intervention_seed": args.seed + index,
        }
        for condition in sorted(RELIABILITY_INTERVENTIONS):
            conditions[condition].append(
                model.generate(
                    frames,
                    sample["prompt"],
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    reliability_intervention=condition,
                    **context,
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
    original = evaluations["predicted"]["metrics"]
    deltas = {
        condition: {
            metric: float(result["metrics"].get(metric, 0.0))
            - float(original.get(metric, 0.0))
            for metric in original
        }
        for condition, result in evaluations.items()
        if condition != "predicted"
    }
    payload = {
        "protocol": {
            "checkpoint": args.checkpoint,
            "base_model": args.base_model,
            "base_model_revision": args.base_model_revision,
            "split": args.split,
            "seed": args.seed,
            "num_samples": limit,
            "fixed_checkpoint": True,
            "fixed_video_input": True,
            "intervened_variable": "appearance_motion_reliability_field_only",
            "interpretation": "decision-pathway intervention, not probability calibration",
        },
        "evaluations": evaluations,
        "delta_from_predicted": deltas,
        "provenance": collect_runtime_metadata(
            data_dir=args.data_dir, checkpoint=args.checkpoint
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(deltas, indent=2))


if __name__ == "__main__":
    main()
