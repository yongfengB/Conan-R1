#!/usr/bin/env python3
"""Collect single-run model and ablation evaluations without inventing values."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = [
    "BLEU-1",
    "BLEU-4",
    "METEOR",
    "ROUGE-L",
    "tIoU",
    "Recall@tIoU=0.3",
    "Recall@tIoU=0.5",
    "Recall@tIoU=0.7",
    "Event-Accuracy",
    "Event-Macro-F1",
    "Parse-Success",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output_json", default="results/final_metrics.json")
    parser.add_argument("--output_csv", default="results/final_metrics.csv")
    args = parser.parse_args()

    rows = []
    for path_text in args.inputs:
        path = Path(path_text)
        payload = json.loads(path.read_text(encoding="utf-8"))
        protocol = payload.get("protocol", {})
        metrics = payload.get("metrics", {})
        missing = [metric for metric in METRICS if metric not in metrics]
        if missing:
            raise ValueError(f"{path} is missing metrics: {missing}")
        rows.append(
            {
                "model": protocol.get("model_name", path.stem),
                "result_file": str(path),
                **{metric: float(metrics[metric]) for metric in METRICS},
            }
        )

    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps({"single_run": True, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    csv_path = Path(args.output_csv)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["model", "result_file", *METRICS]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
