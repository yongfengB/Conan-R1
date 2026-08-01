#!/usr/bin/env python3
"""Compare an evaluation with a version-matched verified release reference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    observed = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    if reference.get("verification_status") != "verified_release":
        raise ValueError(
            "Reference is not marked verified_release; refusing a misleading "
            "reproduction comparison."
        )
    for field in ("code_revision", "annotations_sha256", "splits_sha256"):
        if not reference.get(field):
            raise ValueError(f"Release reference is missing provenance field {field}.")
        observed_value = observed.get("provenance", {}).get(field)
        if observed_value != reference[field]:
            raise ValueError(
                f"Observed {field}={observed_value!r} does not match the "
                f"release reference {reference[field]!r}."
            )

    failed = []
    for metric, expected in reference["metrics"].items():
        actual = observed.get("metrics", {}).get(metric)
        delta = None if actual is None else abs(float(actual) - float(expected))
        passed = delta is not None and delta <= args.tolerance
        print(
            f"{'PASS' if passed else 'FAIL':4s} {metric:22s} "
            f"expected={expected} observed={actual}"
        )
        if not passed:
            failed.append(metric)
    if failed:
        raise SystemExit("Reproduction check failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
