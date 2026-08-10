#!/usr/bin/env python3
"""Create a source-video-level, blinded pairwise human-evaluation package."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_system(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=results/evaluation.json.")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Both system name and result path are required.")
    return name.strip(), Path(path)


def load_rows(path: Path) -> Dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("per_sample", [])
    result = {}
    for row in rows:
        video_id = str(row["video_id"])
        if video_id in result:
            raise ValueError(f"{path}: duplicate video_id {video_id}.")
        if "raw_output" not in row:
            raise ValueError(f"{path}: per_sample row {video_id} lacks raw_output.")
        result[video_id] = row
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system",
        action="append",
        type=parse_system,
        required=True,
        help="Exactly two NAME=evaluation.json inputs.",
    )
    parser.add_argument("--data_dir", default="data/surv_vau")
    parser.add_argument("--num_sources", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="human_evaluation")
    args = parser.parse_args()
    if len(args.system) != 2:
        raise ValueError("The blinded pairwise protocol requires exactly two systems.")
    if args.num_sources < 300:
        raise ValueError("Use at least 300 independent source videos.")

    names = [item[0] for item in args.system]
    if len(set(names)) != 2:
        raise ValueError("System names must be distinct.")
    predictions = {
        name: load_rows(path) for name, path in args.system
    }
    common_video_ids = set.intersection(
        *(set(rows) for rows in predictions.values())
    )
    if not common_video_ids:
        raise ValueError("The two result files have no common videos.")

    by_source: Dict[str, List[str]] = defaultdict(list)
    first_system = predictions[names[0]]
    for video_id in sorted(common_video_ids):
        source_id = str(first_system[video_id].get("source_video_id", ""))
        if not source_id:
            raise ValueError(f"{video_id} lacks source_video_id.")
        by_source[source_id].append(video_id)
    if len(by_source) < args.num_sources:
        raise ValueError(
            f"Requested {args.num_sources} sources, only {len(by_source)} are common."
        )

    rng = random.Random(args.seed)
    strata = defaultdict(list)
    for video_id in sorted(common_video_ids):
        row = first_system[video_id]
        stratum = (
            str(row.get("degradation_domain", "unspecified")),
            str(row.get("degradation_level", "unspecified")),
            str(row.get("ground_truth_event_type", "unspecified")),
        )
        strata[stratum].append(video_id)
    for values in strata.values():
        rng.shuffle(values)

    chosen_video_ids = []
    used_sources = set()
    while len(chosen_video_ids) < args.num_sources:
        made_progress = False
        for stratum in sorted(strata):
            candidates = strata[stratum]
            while candidates:
                video_id = candidates.pop()
                source_id = str(
                    first_system[video_id]["source_video_id"]
                )
                if source_id in used_sources:
                    continue
                chosen_video_ids.append(video_id)
                used_sources.add(source_id)
                made_progress = True
                break
            if len(chosen_video_ids) >= args.num_sources:
                break
        if not made_progress:
            raise RuntimeError(
                "Could not select the requested number of unique sources."
            )

    tasks, key_rows = [], []
    for index, video_id in enumerate(chosen_video_ids):
        source_id = str(first_system[video_id]["source_video_id"])
        reference = first_system[video_id]
        order = list(names)
        rng.shuffle(order)
        task_id = f"HE-{index + 1:04d}"
        tasks.append(
            {
                "task_id": task_id,
                "source_video_id": source_id,
                "video_id": video_id,
                "video_path": str(
                    Path(args.data_dir) / "videos" / f"{video_id}.mp4"
                ),
                "candidate_a": predictions[order[0]][video_id]["raw_output"],
                "candidate_b": predictions[order[1]][video_id]["raw_output"],
            }
        )
        key_rows.append(
            {
                "task_id": task_id,
                "video_id": video_id,
                "source_video_id": source_id,
                "candidate_a_system": order[0],
                "candidate_b_system": order[1],
                "ground_truth_event_type": reference.get(
                    "ground_truth_event_type", ""
                ),
                "ground_truth_interval": reference.get(
                    "ground_truth_interval", []
                ),
                "degradation_level": reference.get("degradation_level"),
                "degradation_domain": reference.get("degradation_domain"),
            }
        )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    task_path = output / "blinded_tasks.jsonl"
    key_path = output / "private_key.jsonl"
    task_payload = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in tasks
    )
    key_payload = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in key_rows
    )
    task_path.write_text(task_payload, encoding="utf-8")
    key_path.write_text(key_payload, encoding="utf-8")
    schema = {
        "task_id": "HE-0001",
        "rater_id": "anonymous-rater-id",
        "human_event_type": "independent event label",
        "human_interval": [0.0, 1.0],
        "candidate_a": {
            "event_correctness": 1,
            "temporal_correctness": 1,
            "explanation_correctness": 1,
            "evidence_groundedness": 1,
            "sufficiency": 1,
            "hallucination": False,
        },
        "candidate_b": {
            "event_correctness": 1,
            "temporal_correctness": 1,
            "explanation_correctness": 1,
            "evidence_groundedness": 1,
            "sufficiency": 1,
            "hallucination": False,
        },
        "pairwise_preference": "a",
    }
    (output / "rating_schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )
    manifest = {
        "seed": args.seed,
        "num_source_videos": len(tasks),
        "systems": names,
        "raters_required_per_task": 3,
        "stratification": [
            "degradation_domain",
            "degradation_level",
            "ground_truth_event_type",
        ],
        "blinded_tasks_sha256": hashlib.sha256(
            task_payload.encode("utf-8")
        ).hexdigest(),
        "private_key_sha256": hashlib.sha256(
            key_payload.encode("utf-8")
        ).hexdigest(),
        "no_teacher_reference_shown_to_raters": True,
        "confidence_intervals": "not requested",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
