#!/usr/bin/env python3
"""Summarize three-rater blinded judgments without bootstrap intervals."""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from training.rewards import compute_rt


SCORE_FIELDS = (
    "event_correctness",
    "temporal_correctness",
    "explanation_correctness",
    "evidence_groundedness",
    "sufficiency",
)


def load_jsonl(path: Path) -> List[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize_label(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def fleiss_kappa(items: Sequence[Sequence[str]]) -> float:
    """Fleiss' kappa for equally rated categorical items."""
    if not items:
        return 0.0
    n = len(items[0])
    if n < 2 or any(len(item) != n for item in items):
        raise ValueError("Fleiss kappa requires an equal number of raters per item.")
    categories = sorted({label for item in items for label in item})
    total = len(items) * n
    proportions = {
        category: sum(item.count(category) for item in items) / total
        for category in categories
    }
    observed = sum(
        (sum(count * count for count in Counter(item).values()) - n)
        / (n * (n - 1))
        for item in items
    ) / len(items)
    expected = sum(value * value for value in proportions.values())
    return 1.0 if expected == 1.0 else (observed - expected) / (1.0 - expected)


def mean(values: Iterable[float]) -> float:
    collected = list(values)
    return float(sum(collected) / len(collected)) if collected else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--private_key", required=True)
    parser.add_argument("--ratings", required=True)
    parser.add_argument("--raters_per_task", type=int, default=3)
    parser.add_argument(
        "--output", default="human_evaluation/summary.json"
    )
    args = parser.parse_args()
    if args.raters_per_task < 3:
        raise ValueError("At least three raters per task are required.")

    task_ids = {row["task_id"] for row in load_jsonl(Path(args.tasks))}
    keys = {
        row["task_id"]: row for row in load_jsonl(Path(args.private_key))
    }
    ratings = load_jsonl(Path(args.ratings))
    if task_ids != set(keys):
        raise ValueError("Blinded tasks and private key do not align.")

    by_task: Dict[str, List[dict]] = defaultdict(list)
    for row in ratings:
        task_id = row.get("task_id")
        if task_id not in task_ids:
            raise ValueError(f"Unknown task_id in ratings: {task_id}.")
        by_task[task_id].append(row)
    missing = sorted(task_ids - set(by_task))
    if missing:
        raise ValueError(f"Ratings are missing for {len(missing)} tasks.")

    selected_by_task = {}
    for task_id, rows in by_task.items():
        rater_ids = [str(row.get("rater_id", "")) for row in rows]
        if (
            "" in rater_ids
            or len(rows) != args.raters_per_task
            or len(set(rater_ids)) != args.raters_per_task
        ):
            raise ValueError(
                f"{task_id} must have exactly {args.raters_per_task} unique raters."
            )
        unique = {}
        for row in rows:
            unique.setdefault(str(row["rater_id"]), row)
        selected_by_task[task_id] = list(unique.values())

    system_scores = defaultdict(lambda: defaultdict(list))
    preference_counts = defaultdict(Counter)
    event_items, interval_agreements = [], []
    ordinal_items = defaultdict(lambda: defaultdict(list))

    for task_id, task_ratings in selected_by_task.items():
        key = keys[task_id]
        event_items.append(
            [
                normalize_label(row["human_event_type"])
                for row in task_ratings
            ]
        )
        intervals = [tuple(map(float, row["human_interval"])) for row in task_ratings]
        interval_agreements.extend(
            compute_rt(first, second)
            for first, second in itertools.combinations(intervals, 2)
        )
        for side in ("a", "b"):
            system = key[f"candidate_{side}_system"]
            for field in SCORE_FIELDS:
                values = [
                    int(row[f"candidate_{side}"][field])
                    for row in task_ratings
                ]
                if any(value < 1 or value > 5 for value in values):
                    raise ValueError(f"{task_id}: {field} must be on a 1--5 scale.")
                system_scores[system][field].extend(values)
                ordinal_items[system][field].append(
                    [str(value) for value in values]
                )
            hallucinations = [
                bool(row[f"candidate_{side}"]["hallucination"])
                for row in task_ratings
            ]
            system_scores[system]["hallucination"].extend(hallucinations)

        for row in task_ratings:
            preference = str(row["pairwise_preference"]).lower()
            if preference not in {"a", "b", "tie"}:
                raise ValueError(
                    f"{task_id}: preference must be a, b, or tie."
                )
            if preference == "tie":
                for system in (
                    key["candidate_a_system"],
                    key["candidate_b_system"],
                ):
                    preference_counts[system]["tie"] += 1
            else:
                winner = key[f"candidate_{preference}_system"]
                loser_side = "b" if preference == "a" else "a"
                loser = key[f"candidate_{loser_side}_system"]
                preference_counts[winner]["win"] += 1
                preference_counts[loser]["loss"] += 1

    systems = {}
    for system, fields in system_scores.items():
        systems[system] = {
            **{
                field: mean(float(value) for value in fields[field])
                for field in SCORE_FIELDS
            },
            "hallucination_rate": mean(
                float(value) for value in fields["hallucination"]
            ),
            "preference": dict(preference_counts[system]),
            "rating_agreement_fleiss_kappa": {
                field: fleiss_kappa(ordinal_items[system][field])
                for field in SCORE_FIELDS
            },
        }

    summary = {
        "num_tasks": len(selected_by_task),
        "raters_per_task": args.raters_per_task,
        "systems": systems,
        "independent_event_label_fleiss_kappa": fleiss_kappa(event_items),
        "mean_pairwise_human_interval_tIoU": mean(interval_agreements),
        "confidence_intervals": "not computed by request",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
