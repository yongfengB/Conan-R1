"""Audit the redistributable demo data and raw prediction closure."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evaluation.metrics import compute_tiou
from model.parser import extract_temporal_interval, parse_structured_output


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_demo_split_hash_and_source_isolation():
    manifest = json.loads((DEMO / "split_manifest.json").read_text())
    assert sha256(DEMO / "annotations.jsonl") == manifest["annotations_sha256"]
    assert sha256(DEMO / "splits.json") == manifest["splits_sha256"]
    rows = [json.loads(line) for line in (DEMO / "annotations.jsonl").read_text().splitlines()]
    splits = json.loads((DEMO / "splits.json").read_text())
    source_splits = {}
    for row in rows:
        split = splits[row["video_id"]]
        previous = source_splits.setdefault(row["source_video_id"], split)
        assert previous == split


def test_demo_raw_predictions_parse_and_close_tiou():
    rows = {
        row["video_id"]: row
        for row in (
            json.loads(line)
            for line in (DEMO / "annotations.jsonl").read_text().splitlines()
        )
    }
    predictions_path = ROOT / "results" / "demo_raw_predictions.jsonl"
    manifest = json.loads((DEMO / "split_manifest.json").read_text())
    assert sha256(predictions_path) == manifest["raw_predictions_sha256"]
    predictions = [
        json.loads(line) for line in predictions_path.read_text().splitlines()
    ]
    test_ids = {
        video_id
        for video_id, split in json.loads((DEMO / "splits.json").read_text()).items()
        if split == "test"
    }
    assert {row["video_id"] for row in predictions} == test_ids
    for prediction in predictions:
        parsed = parse_structured_output(prediction["raw_output"])
        assert parsed is not None
        interval = extract_temporal_interval(parsed.answer_block)
        reference = rows[prediction["video_id"]]
        assert compute_tiou(
            interval,
            tuple(reference["gt_interval"]),
            reference["duration_sec"],
        ) == pytest.approx(1.0)
