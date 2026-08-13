#!/usr/bin/env python3
"""Recompute retention, normalized drop and robustness AUC from an evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.robustness import (
    summarize_robustness,
    validate_robustness_coverage,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation_json")
    parser.add_argument(
        "--scope", choices=["synthetic", "complete"], default="synthetic"
    )
    parser.add_argument(
        "--output", default="results/robustness_summary.json"
    )
    args = parser.parse_args()
    payload = json.loads(Path(args.evaluation_json).read_text(encoding="utf-8"))
    rows = payload.get("per_sample", [])
    required_domains = (
        ("clean", "synthetic_seen", "synthetic_unseen", "natural")
        if args.scope == "complete"
        else ("clean", "synthetic_seen", "synthetic_unseen")
    )
    validate_robustness_coverage(rows, required_domains=required_domains)
    report = summarize_robustness(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
