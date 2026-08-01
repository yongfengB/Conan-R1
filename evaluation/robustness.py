"""Robustness summaries over degradation levels and domains."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

import numpy as np
from .metrics import compute_corpus_bleu


def validate_robustness_coverage(
    rows: Sequence[Dict],
    required_levels: Iterable[float] = (0.0, 0.2, 0.4, 0.8),
    required_domains: Iterable[str] = (
        "clean",
        "synthetic_seen",
        "synthetic_unseen",
        "natural",
    ),
) -> Dict[str, int]:
    """Require the complete clean/seen/unseen/natural stress-test matrix."""
    level_counts = defaultdict(int)
    domain_counts = defaultdict(int)
    level_sources = defaultdict(set)
    for row in rows:
        level_counts[float(row["degradation_level"])] += 1
        domain_counts[str(row["degradation_domain"])] += 1
        source_id = str(row.get("source_video_id", ""))
        if not source_id:
            raise ValueError("Robustness rows require source_video_id.")
        level_sources[float(row["degradation_level"])].add(source_id)
    missing_levels = [
        float(level) for level in required_levels if level_counts[float(level)] == 0
    ]
    missing_domains = [
        str(domain) for domain in required_domains if domain_counts[str(domain)] == 0
    ]
    if missing_levels or missing_domains:
        raise ValueError(
            "Incomplete robustness coverage. Missing levels="
            f"{missing_levels}, domains={missing_domains}."
        )
    paired_sources = set.intersection(
        *(level_sources[float(level)] for level in required_levels)
    )
    if not paired_sources:
        raise ValueError(
            "No source_video_id is represented at every required degradation level."
        )
    return {
        **{f"level:{key}": value for key, value in level_counts.items()},
        **{f"domain:{key}": value for key, value in domain_counts.items()},
        "paired_level_sources": len(paired_sources),
    }


def _mean(rows: Sequence[Dict], metric: str) -> float:
    values = [float(row[metric]) for row in rows]
    return float(sum(values) / len(values)) if values else 0.0


def _score(rows: Sequence[Dict], metric: str) -> float:
    if metric in {"BLEU-1", "BLEU-4"}:
        return compute_corpus_bleu(
            [row["predicted_answer"] for row in rows],
            [row["ground_truth_answer"] for row in rows],
            max_order=1 if metric == "BLEU-1" else 4,
        )
    return _mean(rows, metric)


def summarize_robustness(
    rows: Sequence[Dict],
    metrics: Iterable[str] = (
        "BLEU-1",
        "BLEU-4",
        "METEOR",
        "ROUGE-L",
        "tIoU",
    ),
) -> Dict:
    by_level = defaultdict(list)
    by_domain = defaultdict(list)
    by_combination = defaultdict(list)
    for row in rows:
        by_level[float(row["degradation_level"])].append(row)
        by_domain[row["degradation_domain"]].append(row)
        by_combination[row["degradation_combination"]].append(row)

    levels = sorted(by_level)
    if 0.0 not in by_level:
        raise ValueError("Robustness analysis requires the 0% reference level.")
    level_source_sets = {
        level: {
            str(row.get("source_video_id", ""))
            for row in subset
            if str(row.get("source_video_id", ""))
        }
        for level, subset in by_level.items()
    }
    paired_sources = set.intersection(
        *(level_source_sets[level] for level in levels)
    )
    if not paired_sources:
        raise ValueError(
            "Level curves require at least one source represented at every level."
        )
    paired_by_level = {
        level: [
            row
            for row in by_level[level]
            if str(row.get("source_video_id", "")) in paired_sources
        ]
        for level in levels
    }
    report = {
        "levels": {},
        "domains": {},
        "combinations": {},
        "summary": {},
        "paired_level_source_count": len(paired_sources),
    }
    for level in levels:
        report["levels"][str(level)] = {
            metric: _score(paired_by_level[level], metric)
            for metric in metrics
        }
    for name, subset in sorted(by_domain.items()):
        report["domains"][name] = {
            metric: _score(subset, metric) for metric in metrics
        }
    for name, subset in sorted(by_combination.items()):
        report["combinations"][name] = {
            metric: _score(subset, metric) for metric in metrics
        }

    for metric in metrics:
        clean = report["levels"]["0.0"][metric]
        metric_summary = {}
        values = []
        for level in levels:
            score = report["levels"][str(level)][metric]
            retention = score / clean if clean > 0.0 else 0.0
            metric_summary[str(level)] = {
                "score": score,
                "retention": retention,
                "normalized_drop": 1.0 - retention,
            }
            values.append(score)
        max_level = max(levels)
        auc = (
            float(np.trapz(values, levels) / max_level)
            if max_level > 0.0
            else values[0]
        )
        report["summary"][metric] = {
            "by_level": metric_summary,
            "robustness_auc": auc,
            "normalized_auc": auc / clean if clean > 0.0 else 0.0,
        }
    return report
