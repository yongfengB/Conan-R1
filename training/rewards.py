"""Verifiable reward functions used by Conan-R1.

All component rewards are bounded to ``[0, 1]`` before aggregation:

``r_d``
    Degradation-factor recognition and severity agreement.
``r_e``
    Exact agreement with an independently annotated event category.
``r_t``
    Temporal intersection over union.
``r_l``
    Agreement with an explicit target effective reasoning length.

The functions in this module are deliberately independent of a language model
judge.  This prevents the student from being rewarded by the same model family
that generated the free-form annotations.
"""
from __future__ import annotations

import math
import re
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


Profile = Sequence[Tuple[str, float]]


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _tokenize_simple(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _remove_repeated_ngrams(tokens: List[str], n: int) -> List[str]:
    """Remove later occurrences of an exact repeated n-gram."""
    if len(tokens) < n:
        return tokens
    seen = set()
    result: List[str] = []
    index = 0
    while index < len(tokens):
        if index + n <= len(tokens):
            ngram = tuple(tokens[index : index + n])
            if ngram in seen:
                index += n
                continue
            seen.add(ngram)
        result.append(tokens[index])
        index += 1
    return result


def effective_length(text: str) -> int:
    """Return token length after deterministic repeated 3--5-gram removal."""
    tokens = _tokenize_simple(text)
    for ngram_order in (5, 4, 3):
        tokens = _remove_repeated_ngrams(tokens, ngram_order)
    return len(tokens)


def canonicalize_factor(name: str) -> str:
    """Normalize degradation-factor spellings without merging factor classes."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _sanitize_profile(profile: Profile) -> List[Tuple[str, float]]:
    sanitized: List[Tuple[str, float]] = []
    for name, severity in profile:
        factor = canonicalize_factor(str(name))
        if not factor:
            continue
        value = float(severity)
        if not math.isfinite(value):
            continue
        sanitized.append((factor, _clip01(value)))
    return sanitized


def compute_rd(
    pred_profile: Profile,
    gt_profile: Profile,
    lambda_s: float = 0.5,
    lambda_fp: float = 0.3,
    lambda_fn: float = 0.3,
) -> float:
    """Compute bounded degradation-profile agreement.

    Matching is restricted to identical canonical factor names.  Severity
    agreement contributes ``1-lambda_s*absolute_error``.  False-positive and
    false-negative penalties are normalized by the corresponding profile size.
    """
    predicted = _sanitize_profile(pred_profile)
    reference = _sanitize_profile(gt_profile)
    if not predicted and not reference:
        return 1.0
    if not predicted or not reference:
        return 0.0

    pred_by_type = {}
    gt_by_type = {}
    for index, (name, severity) in enumerate(predicted):
        pred_by_type.setdefault(name, []).append((index, severity))
    for index, (name, severity) in enumerate(reference):
        gt_by_type.setdefault(name, []).append((index, severity))

    matched_pred = set()
    matched_gt = set()
    match_quality = 0.0
    for factor_type in sorted(set(pred_by_type) & set(gt_by_type)):
        pred_items = pred_by_type[factor_type]
        gt_items = gt_by_type[factor_type]
        cost = np.zeros((len(pred_items), len(gt_items)), dtype=np.float64)
        for pred_index, (_, pred_severity) in enumerate(pred_items):
            for gt_index, (_, gt_severity) in enumerate(gt_items):
                quality = _clip01(
                    1.0 - lambda_s * abs(pred_severity - gt_severity)
                )
                cost[pred_index, gt_index] = -quality
        row_indices, column_indices = linear_sum_assignment(cost)
        for row_index, column_index in zip(row_indices, column_indices):
            match_quality += -cost[row_index, column_index]
            matched_pred.add(pred_items[row_index][0])
            matched_gt.add(gt_items[column_index][0])

    false_positives = len(predicted) - len(matched_pred)
    false_negatives = len(reference) - len(matched_gt)
    base_score = match_quality / max(len(predicted), len(reference), 1)
    score = (
        base_score
        - lambda_fp * false_positives / max(len(predicted), 1)
        - lambda_fn * false_negatives / max(len(reference), 1)
    )
    return _clip01(score)


def compute_ro(
    pred_profile: Profile,
    gt_profile: Profile,
    lambda_s: float = 0.5,
    lambda_fp: float = 0.3,
    lambda_fn: float = 0.3,
) -> float:
    """Deprecated compatibility alias for :func:`compute_rd`."""
    return compute_rd(pred_profile, gt_profile, lambda_s, lambda_fp, lambda_fn)


def normalize_event_label(label: str) -> str:
    """Normalize a categorical event label for exact comparison."""
    return " ".join(re.findall(r"[a-z0-9]+", str(label).lower()))


def compute_re(
    pred_event: Optional[str],
    gt_event: str,
    aliases: Optional[Iterable[str]] = None,
) -> float:
    """Return exact categorical event correctness.

    ``aliases`` must be supplied by the dataset manifest; no semantic model is
    called at reward time.
    """
    if pred_event is None:
        return 0.0
    predicted = normalize_event_label(pred_event)
    accepted = {normalize_event_label(gt_event)}
    accepted.update(normalize_event_label(alias) for alias in aliases or [])
    accepted.discard("")
    return float(bool(predicted) and predicted in accepted)


def compute_rt(
    pred_interval: Optional[Tuple[float, float]],
    gt_interval: Tuple[float, float],
    duration_sec: Optional[float] = None,
) -> float:
    """Compute temporal IoU after validating both intervals."""
    if pred_interval is None:
        return 0.0
    pred_start, pred_end = map(float, pred_interval)
    gt_start, gt_end = map(float, gt_interval)
    if (
        not all(map(math.isfinite, (pred_start, pred_end, gt_start, gt_end)))
        or pred_start < 0.0
        or gt_start < 0.0
        or pred_start >= pred_end
        or gt_start >= gt_end
        or (
            duration_sec is not None
            and (
                not math.isfinite(float(duration_sec))
                or float(duration_sec) <= 0.0
                or pred_end > float(duration_sec)
                or gt_end > float(duration_sec)
            )
        )
    ):
        return 0.0
    intersection = max(
        0.0, min(pred_end, gt_end) - max(pred_start, gt_start)
    )
    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
    return _clip01(intersection / union) if union > 0.0 else 0.0


def compute_rl(
    pred_reasoning: str,
    target_length: int,
    tolerance: float = 0.20,
) -> float:
    """Score effective reasoning length against an explicit target.

    A tolerance band prevents the reward from enforcing an exact token count.
    ``target_length`` is the deterministic severity-conditioned target stored
    in the dataset manifest; teacher text is never accepted as the target.
    """
    if isinstance(target_length, bool) or not isinstance(target_length, int):
        raise TypeError("target_length must be an integer policy target.")
    target = target_length
    if target <= 0:
        return 1.0 if effective_length(pred_reasoning) == 0 else 0.0

    predicted = effective_length(pred_reasoning)
    allowed_error = max(1.0, tolerance * target)
    error = abs(predicted - target)
    if error <= allowed_error:
        return 1.0
    decay_range = max(float(target) - allowed_error, 1.0)
    return _clip01(1.0 - (error - allowed_error) / decay_range)


def validate_reward_weights(weights: Mapping[str, float]) -> None:
    missing = {"w_d", "w_e", "w_t", "w_l"} - set(weights)
    if missing:
        raise ValueError("Missing reward weights: " + ", ".join(sorted(missing)))
    values = [float(weights[key]) for key in ("w_d", "w_e", "w_t", "w_l")]
    if any(value < 0.0 for value in values):
        raise ValueError("Reward weights must be non-negative.")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-8):
        raise ValueError(f"Reward weights must sum to 1.0, got {sum(values):.8f}.")


def compute_total_reward(
    rd: float,
    re_: float,
    rt: float,
    rl: float,
    w_d: float = 0.25,
    w_e: float = 0.25,
    w_t: float = 0.25,
    w_l: float = 0.25,
) -> float:
    """Aggregate the four bounded rewards with validated weights."""
    weights = {"w_d": w_d, "w_e": w_e, "w_t": w_t, "w_l": w_l}
    validate_reward_weights(weights)
    components = [rd, re_, rt, rl]
    if any(not math.isfinite(float(component)) for component in components):
        raise ValueError("Reward components must be finite.")
    return _clip01(
        w_d * _clip01(rd)
        + w_e * _clip01(re_)
        + w_t * _clip01(rt)
        + w_l * _clip01(rl)
    )
