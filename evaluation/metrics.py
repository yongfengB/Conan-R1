"""Standard and explicitly named evaluation metrics for Conan-R1."""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from training.rewards import compute_rt, normalize_event_label


def _unit_interval(value: float) -> float:
    """Remove harmless floating-point excursions from normalized metrics."""
    return float(max(0.0, min(1.0, value)))


def compute_corpus_bleu(
    hypotheses: Sequence[str], references: Sequence[str], max_order: int
) -> float:
    """SacreBLEU corpus BLEU, returned on a 0--1 scale."""
    if not hypotheses:
        return 0.0
    if len(hypotheses) != len(references):
        raise ValueError("Hypotheses and references must have equal length.")
    from sacrebleu.metrics import BLEU

    metric = BLEU(
        tokenize="13a",
        smooth_method="exp",
        effective_order=True,
        max_ngram_order=max_order,
    )
    return _unit_interval(
        metric.corpus_score(list(hypotheses), [list(references)]).score / 100.0
    )


def compute_bleu(hyp: str, ref: str, n: int = 1) -> float:
    """Compatibility wrapper for one-pair SacreBLEU."""
    return compute_corpus_bleu([hyp], [ref], max_order=n)


def compute_meteor(hyp: str, ref: str) -> float:
    from nltk.translate.meteor_score import meteor_score

    hyp_tokens = hyp.lower().split()
    ref_tokens = ref.lower().split()
    if not hyp_tokens or not ref_tokens:
        return 0.0
    try:
        return _unit_interval(meteor_score([ref_tokens], hyp_tokens))
    except LookupError as error:
        raise RuntimeError(
            "METEOR requires NLTK wordnet and omw-1.4 resources."
        ) from error


def compute_rouge_l(hyp: str, ref: str) -> float:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return _unit_interval(scorer.score(ref, hyp)["rougeL"].fmeasure)


def compute_tiou(
    pred_interval: Optional[Tuple[float, float]],
    gt_interval: Tuple[float, float],
    duration_sec: Optional[float] = None,
) -> float:
    return compute_rt(pred_interval, gt_interval, duration_sec=duration_sec)


def compute_event_metrics(
    predictions: Sequence[Optional[str]],
    references: Sequence[str],
    aliases: Optional[Sequence[Sequence[str]]] = None,
) -> Dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError("Event predictions and references must align.")
    if not references:
        return {"Event-Accuracy": 0.0, "Event-Macro-F1": 0.0}
    gold = [normalize_event_label(value) for value in references]
    alias_rows = aliases or [() for _ in references]
    if len(alias_rows) != len(references):
        raise ValueError("Event aliases and references must align.")
    predicted = []
    for value, canonical, accepted_aliases in zip(
        predictions, gold, alias_rows
    ):
        normalized = normalize_event_label(value or "")
        accepted = {
            normalize_event_label(alias) for alias in accepted_aliases
        }
        accepted.discard("")
        # Manifest-declared aliases are deterministically mapped back to the
        # canonical category before accuracy and macro-F1 are computed.
        predicted.append(canonical if normalized in accepted else normalized)
    accuracy = sum(p == g for p, g in zip(predicted, gold)) / len(gold)
    labels = sorted(set(gold))
    per_label_f1 = []
    for label in labels:
        true_positive = sum(
            p == label and g == label for p, g in zip(predicted, gold)
        )
        false_positive = sum(
            p == label and g != label for p, g in zip(predicted, gold)
        )
        false_negative = sum(
            p != label and g == label for p, g in zip(predicted, gold)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        per_label_f1.append(
            0.0 if denominator == 0 else 2 * true_positive / denominator
        )
    return {
        "Event-Accuracy": float(accuracy),
        "Event-Macro-F1": float(sum(per_label_f1) / max(1, len(per_label_f1))),
    }


def compute_tiou_recalls(
    scores: Sequence[float], thresholds: Iterable[float] = (0.3, 0.5, 0.7)
) -> Dict[str, float]:
    if not scores:
        return {f"Recall@tIoU={threshold:.1f}": 0.0 for threshold in thresholds}
    return {
        f"Recall@tIoU={threshold:.1f}": float(
            sum(score >= threshold for score in scores) / len(scores)
        )
        for threshold in thresholds
    }


def compute_cider(
    hypotheses: Sequence[str],
    references: Sequence[Union[str, Sequence[str]]],
) -> float:
    """Official COCO-caption CIDEr implementation.

    No approximate fallback is provided.  Install ``pycocoevalcap`` or report
    the evaluation as a custom lexical protocol under a different name.
    """
    try:
        from pycocoevalcap.cider.cider import Cider
    except ImportError as error:
        raise RuntimeError(
            "Official CIDEr requires pycocoevalcap; refusing to report the old "
            "simplified approximation as CIDEr."
        ) from error
    if len(hypotheses) != len(references):
        raise ValueError("CIDEr hypotheses and references must align.")
    res = {
        index: [hypothesis]
        for index, hypothesis in enumerate(hypotheses)
    }
    gts = {
        index: (
            list(reference)
            if not isinstance(reference, str)
            else [reference]
        )
        for index, reference in enumerate(references)
    }
    score, _ = Cider().compute_score(gts, res)
    return float(score)


_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = set(string.punctuation)


def normalize_vqa_answer(answer: str) -> str:
    """Apply the standard VQA-style lowercase/article/punctuation normalization."""
    text = str(answer).lower().replace("\n", " ").replace("\t", " ")
    text = "".join(" " if char in _PUNCTUATION else char for char in text)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def compute_vqa_accuracy(
    predictions: Sequence[str],
    references: Sequence[Union[str, Sequence[str]]],
) -> float:
    """VQA consensus accuracy; single-reference cases reduce to normalized EM."""
    if len(predictions) != len(references):
        raise ValueError("VQA predictions and references must align.")
    if not predictions:
        return 0.0
    scores = []
    for prediction, reference in zip(predictions, references):
        gold_answers = (
            [reference] if isinstance(reference, str) else list(reference)
        )
        normalized_prediction = normalize_vqa_answer(prediction)
        matches = sum(
            normalize_vqa_answer(answer) == normalized_prediction
            for answer in gold_answers
        )
        scores.append(min(1.0, matches / 3.0) if len(gold_answers) > 1 else float(matches > 0))
    return float(sum(scores) / len(scores))
