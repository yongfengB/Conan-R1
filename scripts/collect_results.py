#!/usr/bin/env python3
"""Collect single-run model and ablation evaluations without inventing values."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from scripts._common import prediction_rows_sha256


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


def validate_result_artifact(payload: dict, path: Path) -> None:
    """Reject aggregate-only or identity-free numerical artifacts."""
    per_sample = payload.get("per_sample")
    if not isinstance(per_sample, list) or not per_sample:
        raise ValueError(f"{path} has no per-sample raw predictions.")
    if any(
        not isinstance(row.get("raw_output"), str) or not row["raw_output"]
        for row in per_sample
    ):
        raise ValueError(f"{path} contains a row without raw_output.")
    provenance = payload.get("provenance", {})
    required_hashes = (
        "annotations_sha256",
        "splits_sha256",
        "split_manifest_sha256",
        "raw_predictions_sha256",
    )
    missing = [name for name in required_hashes if not provenance.get(name)]
    if missing:
        raise ValueError(f"{path} lacks dataset provenance: {missing}")
    if not re.fullmatch(r"[0-9a-f]{40}", str(provenance.get("code_revision", ""))):
        raise ValueError(
            f"{path} lacks an exact 40-hex Git commit identity."
        )
    if provenance.get("git_worktree_clean") is not True:
        raise ValueError(f"{path} was produced from a dirty or unknown worktree.")
    for name in required_hashes:
        if not re.fullmatch(r"[0-9a-f]{64}", str(provenance.get(name, ""))):
            raise ValueError(f"{path} has an invalid {name}.")
    actual_predictions_hash = prediction_rows_sha256(per_sample)
    if provenance["raw_predictions_sha256"] != actual_predictions_hash:
        raise ValueError(f"{path} raw predictions do not match their SHA256.")
    protocol = payload.get("protocol", {})
    if protocol.get("artifact_role") != "paper_evidence":
        raise ValueError(f"{path} is not declared as paper_evidence.")
    if not protocol.get("checkpoint"):
        raise ValueError(f"{path} lacks the evaluated checkpoint path.")
    for name in ("checkpoint_identity_sha256", "resolved_config_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(provenance.get(name, ""))):
            raise ValueError(f"{path} lacks a valid {name}.")
    components = provenance.get("checkpoint_files_sha256")
    if not isinstance(components, dict) or not components:
        raise ValueError(f"{path} lacks checkpoint component hashes.")


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
        validate_result_artifact(payload, path)
        protocol = payload.get("protocol", {})
        metrics = payload.get("metrics", {})
        missing = [metric for metric in METRICS if metric not in metrics]
        if missing:
            raise ValueError(f"{path} is missing metrics: {missing}")
        rows.append(
            {
                "model": protocol.get("model_name", path.stem),
                "result_file": str(path),
                "code_revision": payload["provenance"]["code_revision"],
                "annotations_sha256": payload["provenance"]["annotations_sha256"],
                "splits_sha256": payload["provenance"]["splits_sha256"],
                "split_manifest_sha256": payload["provenance"]["split_manifest_sha256"],
                "checkpoint_identity_sha256": payload["provenance"].get(
                    "checkpoint_identity_sha256"
                ),
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
            handle,
            fieldnames=[
                "model",
                "result_file",
                "code_revision",
                "annotations_sha256",
                "splits_sha256",
                "split_manifest_sha256",
                "checkpoint_identity_sha256",
                *METRICS,
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
