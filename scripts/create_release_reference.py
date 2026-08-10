#!/usr/bin/env python3
"""Freeze a verified evaluation as a provenance-bound reproduction target."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation")
    parser.add_argument("--output", default="results/release_reference.json")
    args = parser.parse_args()
    payload = json.loads(
        Path(args.evaluation).read_text(encoding="utf-8")
    )
    if not payload.get("protocol", {}).get(
        "complete_robustness_coverage", False
    ):
        raise ValueError(
            "Refusing to freeze a release reference without complete robustness coverage."
        )
    provenance = payload.get("provenance", {})
    required = ("code_revision", "annotations_sha256", "splits_sha256")
    missing = [field for field in required if not provenance.get(field)]
    if missing:
        raise ValueError(f"Evaluation lacks provenance fields: {missing}.")
    if not payload.get("metrics"):
        raise ValueError("Evaluation has no metrics.")
    reference = {
        "verification_status": "verified_release",
        "source_evaluation": args.evaluation,
        "code_revision": provenance["code_revision"],
        "annotations_sha256": provenance["annotations_sha256"],
        "splits_sha256": provenance["splits_sha256"],
        "checkpoint_sha256": provenance.get("checkpoint_sha256"),
        "protocol": payload["protocol"],
        "metrics": payload["metrics"],
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(
            f"{output} already exists; review and remove it explicitly before refreezing."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reference, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
