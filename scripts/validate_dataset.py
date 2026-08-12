#!/usr/bin/env python3
"""Strict validation gate for the version-matched Surv-VAU release."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED_FIELDS = {
    "video_id",
    "source_video_id",
    "source_dataset",
    "prompt",
    "degradation_profile",
    "degradation_level",
    "degradation_domain",
    "degradation_combination",
    "synthesis_applied",
    "degradation_protocol",
    "gt_interval",
    "event_type",
    "event_aliases",
    "task_mask",
    "source_video_file",
    "anchor_indices",
    "motion_pair_indices",
    "motion_elapsed_sec",
    "influence_targets",
    "reasoning_target_length",
    "reasoning_target_source",
    "duration_sec",
    "fps",
    "num_source_frames",
    "type_annotation",
    "influence_annotation",
    "reasoning_annotation",
    "conclusion_annotation",
    "answer_annotation",
}
VALID_SPLITS = {"sft_train", "rl_train", "val", "test"}
VALID_LEVELS = {0.0, 0.2, 0.4, 0.8}
VALID_DOMAINS = {"clean", "synthetic_seen", "synthetic_unseen", "natural"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/surv_vau")
    parser.add_argument("--expect_instances", type=int, default=27647)
    parser.add_argument("--expect_sources", type=int, default=3688)
    parser.add_argument("--check_videos", action="store_true")
    parser.add_argument(
        "--require_robustness_coverage",
        action="store_true",
        help="Require clean, seen-synthetic, unseen-synthetic and natural records",
    )
    parser.add_argument(
        "--report", default="results/dataset_validation.json"
    )
    args = parser.parse_args()

    root = Path(args.data_dir)
    annotations_path = root / "annotations.jsonl"
    split_path = root / "splits.json"
    if not annotations_path.is_file() or not split_path.is_file():
        raise FileNotFoundError(
            "Both annotations.jsonl and splits.json are required."
        )
    split_manifest_path = root / "split_manifest.json"
    if not split_manifest_path.is_file():
        raise FileNotFoundError(
            "split_manifest.json is required to audit split provenance."
        )
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    records = []
    errors = []
    with open(annotations_path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = REQUIRED_FIELDS - record.keys()
            if missing:
                errors.append(
                    f"line {line_number}: missing fields {sorted(missing)}"
                )
                continue
            video_id = str(record["video_id"])
            interval = record["gt_interval"]
            duration = float(record["duration_sec"])
            fps = float(record["fps"])
            source_frames = int(record["num_source_frames"])
            if (
                not isinstance(interval, list)
                or len(interval) != 2
                or not (0.0 <= float(interval[0]) < float(interval[1]) <= duration)
            ):
                errors.append(
                    f"line {line_number}: gt_interval must be inside duration_sec"
                )
            if (
                not math.isfinite(duration)
                or not math.isfinite(fps)
                or duration <= 0.0
                or fps <= 0.0
                or source_frames <= 1
                or duration + 1e-6 < (source_frames - 1) / fps
            ):
                errors.append(f"line {line_number}: inconsistent video metadata")
            if float(record["degradation_level"]) not in VALID_LEVELS:
                errors.append(
                    f"line {line_number}: invalid degradation_level"
                )
            if record["degradation_domain"] not in VALID_DOMAINS:
                errors.append(
                    f"line {line_number}: invalid degradation_domain"
                )
            synthetic = record["degradation_domain"].startswith("synthetic_")
            if bool(record["synthesis_applied"]) != synthetic:
                errors.append(
                    f"line {line_number}: synthesis_applied must match the domain"
                )
            if record["degradation_domain"] in {"clean", "natural"} and (
                float(record["degradation_level"]) != 0.0
                or record["degradation_profile"]
                or str(record["degradation_protocol"]) != "source_observation"
            ):
                errors.append(
                    f"line {line_number}: source-observation domain has synthetic fields"
                )
            if synthetic and (
                float(record["degradation_level"]) == 0.0
                or not record["degradation_profile"]
                or str(record["degradation_protocol"]) != "surv-vau-degradation-v1"
            ):
                errors.append(
                    f"line {line_number}: synthetic domain has inconsistent protocol fields"
                )
            if not str(record["event_type"]).strip():
                errors.append(f"line {line_number}: empty event_type")
            if int(record["reasoning_target_length"]) <= 0:
                errors.append(
                    f"line {line_number}: reasoning_target_length must be positive"
                )
            if not isinstance(record["event_aliases"], list):
                errors.append(f"line {line_number}: event_aliases must be a list")
            task_mask = record["task_mask"]
            if (
                not isinstance(task_mask, dict)
                or set(task_mask) != {"event", "temporal"}
                or not any(bool(value) for value in task_mask.values())
            ):
                errors.append(f"line {line_number}: invalid task_mask")
            anchors = record["anchor_indices"]
            motion_pairs = record["motion_pair_indices"]
            elapsed = record["motion_elapsed_sec"]
            if not (
                isinstance(anchors, list)
                and len(anchors) == 25
                and anchors == sorted(anchors)
                and len(set(anchors)) == 25
            ):
                errors.append(f"line {line_number}: invalid anchor_indices")
            if not (
                isinstance(motion_pairs, list)
                and len(motion_pairs) == 25
                and all(
                    isinstance(pair, list)
                    and len(pair) == 2
                    and int(pair[0]) == int(anchor)
                    and int(pair[1]) >= int(pair[0])
                    for pair, anchor in zip(motion_pairs, anchors)
                )
            ):
                errors.append(f"line {line_number}: invalid motion_pair_indices")
            if not (
                isinstance(elapsed, list)
                and len(elapsed) == 25
                and all(float(value) > 0.0 for value in elapsed)
            ):
                errors.append(f"line {line_number}: invalid motion_elapsed_sec")
            influence = record["influence_targets"]
            if not (
                isinstance(influence, dict)
                and set(influence)
                == {"affected_interval", "evidence_branch", "reliability_level", "cue_impact"}
                and influence["evidence_branch"] in {"appearance", "motion", "both"}
                and 0.0 <= float(influence["reliability_level"]) <= 1.0
            ):
                errors.append(f"line {line_number}: invalid influence_targets")
            profile = record["degradation_profile"]
            if not isinstance(profile, list):
                errors.append(
                    f"line {line_number}: degradation_profile must be a list"
                )
            else:
                for factor in profile:
                    if (
                        not isinstance(factor, (list, tuple))
                        or len(factor) != 2
                        or not str(factor[0]).strip()
                        or not 0.0 <= float(factor[1]) <= 1.0
                    ):
                        errors.append(
                            f"line {line_number}: invalid degradation factor"
                        )
                        break
            if video_id not in splits:
                errors.append(f"line {line_number}: missing split assignment")
            records.append(record)

    if errors:
        preview = "\n".join(errors[:20])
        raise ValueError(
            f"Dataset validation found {len(errors)} errors:\n{preview}"
        )

    ids = [record["video_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate video_id values found.")
    if set(ids) != set(splits):
        raise ValueError("annotations.jsonl and splits.json contain different IDs.")
    invalid_splits = set(splits.values()) - VALID_SPLITS
    if invalid_splits:
        raise ValueError(f"Unknown split labels: {sorted(invalid_splits)}")

    source_to_splits = defaultdict(set)
    for record in records:
        source_to_splits[record["source_video_id"]].add(
            splits[record["video_id"]]
        )
    leaked = [
        source for source, assigned in source_to_splits.items() if len(assigned) > 1
    ]
    if leaked:
        raise ValueError(
            f"Source-video split leakage detected for {len(leaked)} sources."
        )
    source_split_counts = Counter(
        next(iter(assigned)) for assigned in source_to_splits.values()
    )
    source_total = sum(source_split_counts.values())
    train_source_total = (
        source_split_counts["sft_train"] + source_split_counts["rl_train"]
    )
    outer_train_fraction = train_source_total / max(1, source_total)
    val_fraction = source_split_counts["val"] / max(1, source_total)
    test_fraction = source_split_counts["test"] / max(1, source_total)
    if not (
        0.67 <= outer_train_fraction <= 0.73
        and 0.12 <= val_fraction <= 0.18
        and 0.12 <= test_fraction <= 0.18
    ):
        raise ValueError(
            "Source-level outer split is not approximately 70/15/15: "
            f"train={outer_train_fraction:.4f}, val={val_fraction:.4f}, "
            f"test={test_fraction:.4f}."
        )

    if len(records) != args.expect_instances:
        raise ValueError(
            f"Expected {args.expect_instances} instances, found {len(records)}."
        )
    if len(source_to_splits) != args.expect_sources:
        raise ValueError(
            f"Expected {args.expect_sources} sources, found {len(source_to_splits)}."
        )
    missing_videos = []
    if args.check_videos:
        missing_videos = [
            video_id
            for video_id in ids
            if not (root / "videos" / f"{video_id}.mp4").is_file()
        ]
        if missing_videos:
            raise ValueError(f"Missing {len(missing_videos)} video files.")

    training_counts = Counter(
        value
        for value in splits.values()
        if value in {"sft_train", "rl_train"}
    )
    training_total = sum(training_counts.values())
    sft_fraction = (
        training_counts["sft_train"] / training_total if training_total else 0.0
    )
    if not 0.27 <= sft_fraction <= 0.33:
        raise ValueError(
            f"SFT fraction is {sft_fraction:.4f}; expected approximately 0.30."
        )
    sft_source_fraction = source_split_counts["sft_train"] / max(
        1, train_source_total
    )
    if not 0.27 <= sft_source_fraction <= 0.33:
        raise ValueError(
            "Source-level SFT fraction is "
            f"{sft_source_fraction:.4f}; expected approximately 0.30."
        )
    domain_counts = Counter(r["degradation_domain"] for r in records)
    level_counts = Counter(float(r["degradation_level"]) for r in records)
    if args.require_robustness_coverage:
        test_records = [
            record for record in records if splits[record["video_id"]] == "test"
        ]
        test_domain_counts = Counter(
            record["degradation_domain"] for record in test_records
        )
        test_level_counts = Counter(
            float(record["degradation_level"])
            for record in test_records
            if record["degradation_domain"] == "synthetic_seen"
            or (
                record["degradation_domain"] == "clean"
                and float(record["degradation_level"]) == 0.0
            )
        )
        missing_domains = sorted(
            VALID_DOMAINS - set(test_domain_counts)
        )
        missing_levels = sorted(VALID_LEVELS - set(test_level_counts))
        if missing_domains or missing_levels:
            raise ValueError(
                "Incomplete robustness coverage: "
                f"missing domains={missing_domains}, levels={missing_levels}."
            )
        test_level_sources = defaultdict(set)
        for record in test_records:
            if record["degradation_domain"] == "synthetic_seen" or (
                record["degradation_domain"] == "clean"
                and float(record["degradation_level"]) == 0.0
            ):
                test_level_sources[float(record["degradation_level"])].add(
                    str(record["source_video_id"])
                )
        paired_test_sources = set.intersection(
            *(test_level_sources[level] for level in sorted(VALID_LEVELS))
        )
        if not paired_test_sources:
            raise ValueError(
                "No test source is represented at every required degradation level."
            )
        held_out_leakage = [
            record["video_id"]
            for record in records
            if record["degradation_domain"] in {"synthetic_unseen", "natural"}
            and splits[record["video_id"]] != "test"
        ]
        if held_out_leakage:
            raise ValueError(
                "Unseen/natural robustness records leaked into training: "
                f"{len(held_out_leakage)} instances."
            )

    split_manifest = json.loads(
        split_manifest_path.read_text(encoding="utf-8")
    )
    expected_annotation_hash = split_manifest.get("annotations_sha256")
    expected_split_hash = split_manifest.get("splits_sha256")
    if expected_annotation_hash != sha256(annotations_path):
        raise ValueError("split_manifest annotations_sha256 does not match.")
    if expected_split_hash != sha256(split_path):
        raise ValueError("split_manifest splits_sha256 does not match.")

    report = {
        "status": "valid",
        "instances": len(records),
        "source_videos": len(source_to_splits),
        "split_instances": dict(Counter(splits.values())),
        "split_sources": dict(source_split_counts),
        "sft_fraction_within_train": sft_fraction,
        "sft_source_fraction_within_train": sft_source_fraction,
        "event_types": dict(Counter(r["event_type"] for r in records)),
        "degradation_levels": {
            str(key): value for key, value in level_counts.items()
        },
        "degradation_domains": dict(domain_counts),
        "annotations_sha256": sha256(annotations_path),
        "splits_sha256": sha256(split_path),
        "videos_checked": args.check_videos,
        "robustness_coverage_required": args.require_robustness_coverage,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
