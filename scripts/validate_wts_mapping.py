#!/usr/bin/env python3
"""Validate a frozen mapping to the official WTS question/answer protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


REQUIRED = {
    "video_id",
    "official_question_id",
    "official_split",
    "answer_references",
    "source_video_id",
    "source_dataset",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/wts_official")
    parser.add_argument("--check_videos", action="store_true")
    parser.add_argument(
        "--output", default="results/wts_mapping_validation.json"
    )
    args = parser.parse_args()
    root = Path(args.data_dir)
    annotations = root / "annotations.jsonl"
    rows = [
        json.loads(line)
        for line in annotations.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    errors = []
    question_ids = []
    for index, row in enumerate(rows, 1):
        missing = REQUIRED - row.keys()
        if missing:
            errors.append(f"line {index}: missing {sorted(missing)}")
            continue
        question_ids.append(str(row["official_question_id"]))
        references = row["answer_references"]
        if not isinstance(references, list) or not references:
            errors.append(f"line {index}: answer_references must be non-empty")
        if args.check_videos and not (
            root / "videos" / f"{row['video_id']}.mp4"
        ).is_file():
            errors.append(f"line {index}: missing video {row['video_id']}")
    if len(question_ids) != len(set(question_ids)):
        errors.append("official_question_id values are not unique")
    if errors:
        raise ValueError("\n".join(errors[:20]))
    report = {
        "status": "valid",
        "instances": len(rows),
        "official_splits": dict(
            Counter(str(row["official_split"]) for row in rows)
        ),
        "annotations_sha256": sha256(annotations),
        "videos_checked": args.check_videos,
        "official_llm_scorer_required_separately": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
