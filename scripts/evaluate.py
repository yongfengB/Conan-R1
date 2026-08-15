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
from model.reliability_pathway import ReliabilityPathwayConfig
from scripts._common import (
    collect_runtime_metadata,
    load_core_protocol,
    prediction_rows_sha256,
    require_dataset,
    resolve_device,
    seed_everything,
)


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
    parser.add_argument(
        "--robustness_scope",
        choices=["none", "synthetic", "complete"],
        default="synthetic",
        help=(
            "synthetic requires clean/seen/unseen protocol coverage; complete "
            "additionally requires a separate natural source-observation partition"
        ),
    )
    parser.add_argument("--output", default="results/evaluation.json")
    parser.add_argument(
        "--table_id",
        choices=[f"Table {index}" for index in range(1, 8)],
        default=None,
        help="Declare paper evidence only when binding this run to a table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.table_id and not args.checkpoint:
        raise ValueError("Paper evidence requires --checkpoint identity.")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    require_dataset(args.data_dir)
    seed_everything(args.seed)
    core_protocol = (
        load_core_protocol(Path(args.checkpoint))
        if args.checkpoint and args.checkpoint_type == "core"
        else None
    )
    if core_protocol is not None:
        preprocessing = core_protocol["motion_preprocessing"]
        if args.num_frames != int(preprocessing["anchors"]):
            raise ValueError("--num_frames must match the checkpoint motion protocol.")
        if args.frame_size != int(preprocessing["frame_size"][0]):
            raise ValueError("--frame_size must match the checkpoint motion protocol.")
    dataset = SurvVAUDataset(
        args.data_dir,
        args.split,
        args.num_frames,
        args.frame_size,
        motion_native_offset=(
            int(core_protocol["motion_preprocessing"]["native_frame_offset"])
            if core_protocol is not None
            else 1
        ),
    )
    if not dataset:
        raise RuntimeError(f"The {args.split} split contains no samples.")
    model = ConanR1Model(
        args.base_model,
        base_model_revision=args.base_model_revision,
        device=resolve_device(args.device),
        enable_lora=args.checkpoint is not None,
        reliability_config=(
            ReliabilityPathwayConfig(**core_protocol["reliability_config"])
            if core_protocol is not None
            else None
        ),
        motion_v_max=(
            float(core_protocol["motion_v_max"])
            if core_protocol is not None
            else None
        ),
        degradation_factor_names=(
            core_protocol["degradation_factor_names"]
            if core_protocol is not None
            else None
        ),
        motion_flow_parameters=(
            core_protocol["motion_flow_parameters"]
            if core_protocol is not None
            else None
        ),
        motion_frame_size=(
            int(core_protocol["motion_preprocessing"]["frame_size"][0])
            if core_protocol is not None
            else 224
        ),
        motion_native_offset=(
            int(core_protocol["motion_preprocessing"]["native_frame_offset"])
            if core_protocol is not None
            else 1
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
    robustness_enabled = not args.wts and args.robustness_scope != "none"
    coverage = None
    if robustness_enabled:
        required_domains = (
            ("clean", "synthetic_seen", "synthetic_unseen", "natural")
            if args.robustness_scope == "complete"
            else ("clean", "synthetic_seen", "synthetic_unseen")
        )
        coverage = validate_robustness_coverage(
            per_sample, required_domains=required_domains
        )
    robustness = summarize_robustness(per_sample) if robustness_enabled else {}
    result_rows = [
        {**row, "raw_output": raw_output}
        for row, raw_output in zip(per_sample, predictions)
    ]
    provenance = collect_runtime_metadata(
        data_dir=args.data_dir, checkpoint=args.checkpoint
    )
    provenance["raw_predictions_sha256"] = prediction_rows_sha256(result_rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "protocol": {
                    "artifact_role": (
                        "paper_evidence" if args.table_id else "audit_only"
                    ),
                    "table_id": args.table_id,
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
                    "complete_robustness_coverage": (
                        None if args.wts else args.robustness_scope == "complete"
                    ),
                    "robustness_scope": args.robustness_scope,
                },
                "metrics": metrics,
                "robustness_coverage": coverage,
                "robustness": robustness,
                "provenance": provenance,
                "per_sample": result_rows,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
