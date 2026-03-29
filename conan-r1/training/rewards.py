"""Reward functions for Conan-R1 GRPO training.

Three rewards:
  ro  — observation-difficulty perception (bipartite matching)
  rt  — temporal boundary localization (tIoU)
  rl  — reasoning compactness (effective length alignment)
"""
from __future__ import annotations
import re
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Effective reasoning length  L(·)
# ---------------------------------------------------------------------------

def _tokenize_simple(text: str) -> List[str]:
    """Whitespace tokenizer (no external dependency required)."""
    return text.lower().split()


def _get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


def _remove_repeated_ngrams(tokens: List[str], n: int) -> List[str]:
    """Remove tokens that form a repeated n-gram (keep first occurrence)."""
    if len(tokens) < n:
        return tokens
    seen: set = set()
    result: List[str] = []
    i = 0
    while i < len(tokens):
        if i + n <= len(tokens):
            ngram = tuple(tokens[i: i + n])
            if ngram in seen:
                i += n  # skip the repeated n-gram
                continue
            seen.add(ngram)
        result.append(tokens[i])
        i += 1
    return result


def _sentence_similarity(s1: str, s2: str) -> float:
    """Jaccard similarity between token sets of two sentences."""
    t1 = set(_tokenize_simple(s1))
    t2 = set(_tokenize_simple(s2))
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def _deduplicate_clauses(clauses: List[str], threshold: float = 0.85) -> List[str]:
    """Remove semantically duplicate clauses (Jaccard similarity ≥ threshold)."""
    unique: List[str] = []
    for clause in clauses:
        if not any(_sentence_similarity(clause, u) >= threshold for u in unique):
            unique.append(clause)
    return unique


MAX_CLAUSE_COUNT = 50


def effective_length(text: str) -> int:
    """Compute effective reasoning length after deduplication.

    Steps:
      1. Remove repeated n-grams (n = 3, 4, 5).
      2. Merge semantically duplicate clauses (Jaccard ≥ 0.85).
      3. Truncate to MAX_CLAUSE_COUNT clauses.

    Postcondition: result ≤ len(tokenize(text))
    """
    tokens = _tokenize_simple(text)
    original_len = len(tokens)

    for n in (3, 4, 5):
        tokens = _remove_repeated_ngrams(tokens, n)

    # Split into clauses on sentence boundaries
    rejoined = " ".join(tokens)
    clauses = re.split(r"[.!?;]", rejoined)
    clauses = [c.strip() for c in clauses if c.strip()]

    unique_clauses = _deduplicate_clauses(clauses, threshold=0.85)
    unique_clauses = unique_clauses[:MAX_CLAUSE_COUNT]

    result = len(" ".join(unique_clauses).split())
    # Guarantee postcondition
    return min(result, original_len)


# ---------------------------------------------------------------------------
# ro — observation-difficulty reward
# ---------------------------------------------------------------------------

def compute_ro(
    pred_profile: List[Tuple[str, float]],
    gt_profile: List[Tuple[str, float]],
    lambda_s: float = 0.5,
    lambda_fp: float = 0.3,
    lambda_fn: float = 0.3,
) -> float:
    """Compute observation-difficulty reward using bipartite matching.

    Args:
        pred_profile: [(factor_name, severity), ...] predicted.
        gt_profile:   [(factor_name, severity), ...] ground-truth.
        lambda_s:  Severity deviation penalty coefficient.
        lambda_fp: False-positive penalty coefficient.
        lambda_fn: False-negative penalty coefficient.

    Returns:
        ro ≤ 1.0
    """
    if not gt_profile and not pred_profile:
        return 1.0

    n_fp = len(pred_profile)
    n_fn = len(gt_profile)

    if not pred_profile or not gt_profile:
        return -lambda_fp * n_fp - lambda_fn * n_fn

    # Group by factor type for bipartite matching
    pred_by_type: dict = {}
    for i, (name, sev) in enumerate(pred_profile):
        pred_by_type.setdefault(name, []).append((i, sev))

    gt_by_type: dict = {}
    for j, (name, sev) in enumerate(gt_profile):
        gt_by_type.setdefault(name, []).append((j, sev))

    matched_pred: set = set()
    matched_gt: set = set()
    score = 0.0

    for factor_type in set(pred_by_type) & set(gt_by_type):
        preds = pred_by_type[factor_type]
        gts = gt_by_type[factor_type]

        # Build cost matrix (we want to maximize, so negate for linear_sum_assignment)
        cost = np.zeros((len(preds), len(gts)))
        for pi, (_, ps) in enumerate(preds):
            for gi, (_, gs) in enumerate(gts):
                cost[pi, gi] = -(1.0 - lambda_s * abs(ps - gs))

        row_ind, col_ind = linear_sum_assignment(cost)
        for pi, gi in zip(row_ind, col_ind):
            score += -cost[pi, gi]
            matched_pred.add(preds[pi][0])
            matched_gt.add(gts[gi][0])

    n_matched = len(matched_pred)
    n_fp_actual = len(pred_profile) - n_matched
    n_fn_actual = len(gt_profile) - n_matched

    ro = score / max(1, n_matched) - lambda_fp * n_fp_actual - lambda_fn * n_fn_actual
    return float(ro)


# ---------------------------------------------------------------------------
# rt — temporal localization reward (tIoU)
# ---------------------------------------------------------------------------

def compute_rt(
    pred_interval: Optional[Tuple[float, float]],
    gt_interval: Tuple[float, float],
) -> float:
    """Compute temporal IoU between predicted and ground-truth intervals.

    Args:
        pred_interval: (start_sec, end_sec) or None if parsing failed.
        gt_interval:   (start_sec, end_sec) ground-truth.

    Returns:
        rt ∈ [0, 1].
    """
    if pred_interval is None:
        return 0.0

    p_start, p_end = pred_interval
    g_start, g_end = gt_interval

    inter_start = max(p_start, g_start)
    inter_end = min(p_end, g_end)
    intersection = max(0.0, inter_end - inter_start)

    union = (p_end - p_start) + (g_end - g_start) - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


# ---------------------------------------------------------------------------
# rl — reasoning compactness reward
# ---------------------------------------------------------------------------

def compute_rl(pred_reasoning: str, gt_reasoning: str) -> float:
    """Compute reasoning compactness reward.

    rl = 1 - |L(pred) - L(gt)| / max(1, L(gt))

    Args:
        pred_reasoning: Predicted <REASONING> block content.
        gt_reasoning:   Ground-truth <REASONING> annotation.

    Returns:
        rl ∈ [0, 1].
    """
    l_pred = effective_length(pred_reasoning)
    l_gt = effective_length(gt_reasoning)
    rl = 1.0 - abs(l_pred - l_gt) / max(1, l_gt)
    return float(max(0.0, min(1.0, rl)))


# ---------------------------------------------------------------------------
# Total reward
# ---------------------------------------------------------------------------

def compute_total_reward(
    ro: float,
    rt: float,
    rl: float,
    alpha: float = 0.4,
    beta: float = 0.4,
    gamma: float = 0.2,
) -> float:
    """Weighted linear combination: R = α*ro + β*rt + γ*rl.

    Weights must sum to 1.0 (α=0.4, β=0.4, γ=0.2 by default).
    """
    return alpha * ro + beta * rt + gamma * rl
