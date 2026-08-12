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
        domain = str(row["degradation_domain"])
        level = float(row["degradation_level"])
        domain_counts[domain] += 1
        synthetic = domain.startswith("synthetic_")
        if bool(row.get("synthesis_applied", False)) != synthetic:
            raise ValueError(
                f"Domain {domain!r} is inconsistent with synthesis_applied."
            )
        if domain == "natural" and row.get("degradation_protocol") != "source_observation":
            raise ValueError(
                "Natural robustness rows must be source observations, not operator outputs."
            )
        source_id = str(row.get("source_video_id", ""))
        if not source_id:
            raise ValueError("Robustness rows require source_video_id.")
        if (domain == "clean" and level == 0.0) or domain == "synthetic_seen":
            level_counts[level] += 1
            level_sources[level].add(source_id)
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
        **{
            f"synthetic_seen_level:{key}": value
            for key, value in level_counts.items()
        },
        **{f"domain:{key}": value for key, value in domain_counts.items()},
        "paired_level_sources": len(paired_sources),
    }


def _mean(rows: Sequence[Dict], metric: str) -> float:
    values = [
        float(row[metric]) for row in rows if row.get(metric) is not None
    ]
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
    synthetic_seen_by_level = defaultdict(list)
    by_domain = defaultdict(list)
    by_combination = defaultdict(list)
    for row in rows:
        level = float(row["degradation_level"])
        domain = str(row["degradation_domain"])
        if (domain == "clean" and level == 0.0) or domain == "synthetic_seen":
            synthetic_seen_by_level[level].append(row)
        by_domain[domain].append(row)
        by_combination[row["degradation_combination"]].append(row)

    levels = sorted(synthetic_seen_by_level)
    if 0.0 not in synthetic_seen_by_level:
        raise ValueError("Synthetic-seen analysis requires a clean 0% reference.")
    level_source_sets = {
        level: {
            str(row.get("source_video_id", ""))
            for row in subset
            if str(row.get("source_video_id", ""))
        }
        for level, subset in synthetic_seen_by_level.items()
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
            for row in synthetic_seen_by_level[level]
            if str(row.get("source_video_id", "")) in paired_sources
        ]
        for level in levels
    }
    report = {
        "synthetic_seen_severity": {},
        "domains": {},
        "combinations": {},
        "summary": {},
        "paired_level_source_count": len(paired_sources),
    }
    for level in levels:
        report["synthetic_seen_severity"][str(level)] = {
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
        clean = report["synthetic_seen_severity"]["0.0"][metric]
        metric_summary = {}
        values = []
        for level in levels:
            score = report["synthetic_seen_severity"][str(level)][metric]
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
