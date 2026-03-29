"""Evaluation metrics for Conan-R1: BLEU, METEOR, ROUGE-L, tIoU, CIDEr, VQA."""
from __future__ import annotations
import collections
import math
from typing import Dict, List, Optional, Tuple

import nltk
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------

def _ngram_counts(tokens: List[str], n: int) -> Dict[Tuple, int]:
    counts: Dict[Tuple, int] = collections.Counter()
    for i in range(len(tokens) - n + 1):
        counts[tuple(tokens[i: i + n])] += 1
    return counts


def compute_bleu(hyp: str, ref: str, n: int = 1) -> float:
    """Compute BLEU-n score for a single hypothesis/reference pair.

    Args:
        hyp: Hypothesis string.
        ref: Reference string.
        n: n-gram order (1 or 4).

    Returns:
        BLEU-n score in [0, 1].
    """
    hyp_tokens = hyp.lower().split()
    ref_tokens = ref.lower().split()

    if not hyp_tokens:
        return 0.0

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(1, len(hyp_tokens))))

    log_score = 0.0
    for k in range(1, n + 1):
        hyp_counts = _ngram_counts(hyp_tokens, k)
        ref_counts = _ngram_counts(ref_tokens, k)

        clipped = sum(min(c, ref_counts.get(ng, 0)) for ng, c in hyp_counts.items())
        total = max(1, len(hyp_tokens) - k + 1)

        if clipped == 0:
            return 0.0
        log_score += math.log(clipped / total)

    return bp * math.exp(log_score / n)


# ---------------------------------------------------------------------------
# METEOR
# ---------------------------------------------------------------------------

def compute_meteor(hyp: str, ref: str) -> float:
    """Compute METEOR score for a single hypothesis/reference pair."""
    try:
        hyp_tokens = hyp.lower().split()
        ref_tokens = ref.lower().split()
        if not hyp_tokens or not ref_tokens:
            return 0.0
        return float(meteor_score([ref_tokens], hyp_tokens))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------

def compute_rouge_l(hyp: str, ref: str) -> float:
    """Compute ROUGE-L F1 score."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = scorer.score(ref, hyp)
    return float(scores["rougeL"].fmeasure)


# ---------------------------------------------------------------------------
# tIoU
# ---------------------------------------------------------------------------

def compute_tiou(
    pred_interval: Optional[Tuple[float, float]],
    gt_interval: Tuple[float, float],
) -> float:
    """Compute temporal IoU.

    Args:
        pred_interval: (start, end) or None.
        gt_interval:   (start, end) ground-truth.

    Returns:
        tIoU ∈ [0, 1].
    """
    if pred_interval is None:
        return 0.0
    p_s, p_e = pred_interval
    g_s, g_e = gt_interval
    inter = max(0.0, min(p_e, g_e) - max(p_s, g_s))
    union = (p_e - p_s) + (g_e - g_s) - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


# ---------------------------------------------------------------------------
# CIDEr (simplified corpus-level)
# ---------------------------------------------------------------------------

def compute_cider(hyps: List[str], refs: List[str]) -> float:
    """Compute a simplified CIDEr score over a corpus.

    Uses TF-IDF weighted n-gram cosine similarity (n=1..4).
    """
    if not hyps or not refs:
        return 0.0

    n_max = 4
    scores = []

    # Build IDF from references
    doc_freq: Dict[Tuple, int] = collections.Counter()
    for ref in refs:
        tokens = ref.lower().split()
        for n in range(1, n_max + 1):
            seen = set(_ngram_counts(tokens, n).keys())
            for ng in seen:
                doc_freq[ng] += 1

    N = len(refs)

    for hyp, ref in zip(hyps, refs):
        hyp_tokens = hyp.lower().split()
        ref_tokens = ref.lower().split()
        sim = 0.0
        for n in range(1, n_max + 1):
            h_counts = _ngram_counts(hyp_tokens, n)
            r_counts = _ngram_counts(ref_tokens, n)
            all_ngs = set(h_counts) | set(r_counts)
            h_vec, r_vec = [], []
            for ng in all_ngs:
                idf = math.log((N + 1) / (doc_freq.get(ng, 0) + 1))
                h_vec.append(h_counts.get(ng, 0) * idf)
                r_vec.append(r_counts.get(ng, 0) * idf)
            dot = sum(a * b for a, b in zip(h_vec, r_vec))
            norm_h = math.sqrt(sum(a ** 2 for a in h_vec)) + 1e-9
            norm_r = math.sqrt(sum(b ** 2 for b in r_vec)) + 1e-9
            sim += dot / (norm_h * norm_r)
        scores.append(sim / n_max)

    return float(sum(scores) / len(scores))


# ---------------------------------------------------------------------------
# VQA accuracy
# ---------------------------------------------------------------------------

def compute_vqa_accuracy(preds: List[str], gts: List[str]) -> float:
    """Compute exact-match VQA accuracy (case-insensitive, stripped)."""
    if not preds:
        return 0.0
    correct = sum(
        p.strip().lower() == g.strip().lower() for p, g in zip(preds, gts)
    )
    return float(correct / len(preds))
