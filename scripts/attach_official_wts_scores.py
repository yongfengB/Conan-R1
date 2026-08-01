#!/usr/bin/env python3
"""Attach externally produced official WTS scorer output with provenance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--official_scores", required=True)
    parser.add_argument("--scorer_commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    evaluation = json.loads(
        Path(args.evaluation).read_text(encoding="utf-8")
    )
    scores = json.loads(
        Path(args.official_scores).read_text(encoding="utf-8")
    )
    if not scores:
        raise ValueError("Official scorer output is empty.")
    evaluation["official_wts_evaluation"] = {
        "scorer_commit": args.scorer_commit,
        "scores": scores,
        "source_file": args.official_scores,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
