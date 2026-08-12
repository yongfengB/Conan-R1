#!/usr/bin/env python3
"""Greedy Conan-R1 evaluation using the paper protocol."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset.dataset import SurvVAUDataset
from evaluation.evaluator import Evaluator
from evaluation.robustness import (
    summarize_robustness,
    validate_robustness_coverage,
)
from model.conan_r1 import ConanR1Model
from scripts._common import (
    collect_runtime_metadata,
    load_core_protocol,
    require_dataset,
    resolve_device,
    seed_everything,
)
from scripts._common import load_config
from scripts.train_sft import build_reliability_config, load_motion_vmax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Conan-R1")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="LoRA checkpoint; omit for the unadapted base model",
    )
    parser.add_argument(
        "--checkpoint_type",
        choices=["core", "lora"],
        default="core",
        help="Use lora for the Structured-SFT baseline; core for Conan-R1",
    )
    parser.add_argument("--model_name", default="Conan-R1")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--wts", action="store_true", help="Include CIDEr/VQA metrics")
    parser.add_argument("--output", default="results/evaluation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    require_dataset(args.data_dir)
    seed_everything(args.seed)
    dataset = SurvVAUDataset(
        args.data_dir, args.split, args.num_frames, args.frame_size
    )
    if not dataset:
        raise RuntimeError(f"The {args.split} split contains no samples.")

    model = ConanR1Model(
        args.base_model,
        base_model_revision=args.base_model_revision,
        device=resolve_device(args.device),
        enable_lora=args.checkpoint is not None,
        reliability_config=(
            build_reliability_config(
                load_config("configs/grpo_config.yaml"), args.base_model
            )
            if args.checkpoint and args.checkpoint_type == "core"
            else None
        ),
        motion_v_max=(
            float(load_core_protocol(Path(args.checkpoint))["motion_v_max"])
            if args.checkpoint and args.checkpoint_type == "core"
            else None
        ),
        degradation_factor_names=(
            load_config(
                load_config("configs/grpo_config.yaml")["model"]["method_config"]
            )["degradation_factors"]
            if args.checkpoint and args.checkpoint_type == "core"
            else None
        ),
    )
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        if args.checkpoint_type == "core":
            model.load_core(args.checkpoint, is_trainable=False)
        else:
            model.load_lora(args.checkpoint, is_trainable=False)
    predictions = []
    references = []
    for sample in dataset:
        frames_tensor = sample["frames"]
        frames_np = (
            frames_tensor.permute(0, 2, 3, 1).numpy() * 255
        ).astype("uint8")
        frames = [frames_np[index] for index in range(frames_np.shape[0])]
        motion_np = (
            sample["motion_frames"].permute(0, 2, 3, 1).numpy() * 255
        ).astype("uint8")
        motion_frames = [motion_np[index] for index in range(motion_np.shape[0])]
        predictions.append(
            model.generate(
                frames,
                sample["prompt"],
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                **(
                    {
                        "motion_frames": motion_frames,
                        "elapsed_seconds": sample["motion_elapsed_sec"],
                        "timestamps": sample["anchor_timestamps_sec"],
                    }
                    if args.checkpoint and args.checkpoint_type == "core"
                    else {}
                ),
            )
        )
        references.append(sample)

    metrics, per_sample = Evaluator().evaluate(
        predictions, references, include_wts_metrics=args.wts
    )
    coverage = None if args.wts else validate_robustness_coverage(per_sample)
    robustness = {} if args.wts else summarize_robustness(per_sample)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "protocol": {
                    "model_name": args.model_name,
                    "checkpoint": args.checkpoint,
                    "checkpoint_type": args.checkpoint_type,
                    "base_model": args.base_model,
                    "base_model_revision": args.base_model_revision,
                    "split": args.split,
                    "num_frames": args.num_frames,
                    "frame_size": args.frame_size,
                    "max_new_tokens": args.max_new_tokens,
                    "decoding": "greedy",
                    "seed": args.seed,
                    "complete_robustness_coverage": None if args.wts else True,
                },
                "metrics": metrics,
                "robustness_coverage": coverage,
                "robustness": robustness,
                "provenance": collect_runtime_metadata(
                    data_dir=args.data_dir, checkpoint=args.checkpoint
                ),
                "per_sample": [
                    {**row, "raw_output": raw_output}
                    for row, raw_output in zip(per_sample, predictions)
                ],
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
