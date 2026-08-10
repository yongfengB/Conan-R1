#!/usr/bin/env python3
"""Score saved raw outputs with the same evaluator used for Conan-R1."""
from __future__ import annotations

import argparse
import json
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
from scripts._common import collect_runtime_metadata, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_jsonl")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--data_dir", default="data/surv_vau")
    parser.add_argument("--split", default="test")
    parser.add_argument("--wts", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    prediction_map = {}
    with open(args.predictions_jsonl, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                prediction_map[row["video_id"]] = row["raw_output"]
    dataset = SurvVAUDataset(
        args.data_dir, args.split, require_videos=False
    )
    references = [dataset[index] for index in range(len(dataset))]
    missing = [
        sample["video_id"]
        for sample in references
        if sample["video_id"] not in prediction_map
    ]
    if missing:
        raise ValueError(f"Missing predictions for {len(missing)} videos.")
    predictions = [
        prediction_map[sample["video_id"]] for sample in references
    ]
    metrics, details = Evaluator().evaluate(
        predictions, references, include_wts_metrics=args.wts
    )
    coverage = None if args.wts else validate_robustness_coverage(details)
    payload = {
        "protocol": {
            "model_name": args.model_name or Path(args.predictions_jsonl).stem,
            "prediction_file": args.predictions_jsonl,
            "split": args.split,
            "decoding": "provided by external system; must be documented",
        },
        "metrics": metrics,
        "robustness": {} if args.wts else summarize_robustness(details),
        "robustness_coverage": coverage,
        "per_sample": [
            {**row, "raw_output": raw_output}
            for row, raw_output in zip(details, predictions)
        ],
        "provenance": {
            **collect_runtime_metadata(data_dir=args.data_dir),
            "predictions_sha256": sha256_file(
                Path(args.predictions_jsonl)
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
