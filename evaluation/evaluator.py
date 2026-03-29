"""Evaluator: runs all metrics on a set of model predictions."""
from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple

from model.parser import extract_temporal_interval, parse_structured_output
from .metrics import (
    compute_bleu,
    compute_cider,
    compute_meteor,
    compute_rouge_l,
    compute_tiou,
    compute_vqa_accuracy,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates Conan-R1 predictions against ground-truth samples.

    Only the <ANSWER> block is used for quantitative scoring.
    """

    def evaluate(
        self,
        predictions: List[str],
        references: List[Dict],
        include_wts_metrics: bool = False,
    ) -> Dict[str, float]:
        """Compute all metrics.

        Args:
            predictions: List of raw model output strings.
            references:  List of dicts with keys:
                           answer_annotation, gt_interval, reasoning_annotation.
            include_wts_metrics: If True, also compute CIDEr and VQA accuracy.

        Returns:
            Dict of metric_name → score.
        """
        bleu1_scores, bleu4_scores, meteor_scores = [], [], []
        rouge_scores, tiou_scores = [], []
        hyp_answers, ref_answers = [], []

        for pred_text, ref in zip(predictions, references):
            parsed = parse_structured_output(pred_text)
            if parsed is None:
                hyp_answer = ""
                pred_interval = None
            else:
                hyp_answer = parsed.answer_block
                pred_interval = extract_temporal_interval(parsed.answer_block)

            gt_answer = ref.get("answer_annotation", "")
            gt_interval = tuple(ref.get("gt_interval", [0.0, 1.0]))

            bleu1_scores.append(compute_bleu(hyp_answer, gt_answer, n=1))
            bleu4_scores.append(compute_bleu(hyp_answer, gt_answer, n=4))
            meteor_scores.append(compute_meteor(hyp_answer, gt_answer))
            rouge_scores.append(compute_rouge_l(hyp_answer, gt_answer))
            tiou_scores.append(compute_tiou(pred_interval, gt_interval))

            hyp_answers.append(hyp_answer)
            ref_answers.append(gt_answer)

        def _mean(lst):
            return sum(lst) / max(1, len(lst))

        results = {
            "BLEU-1": _mean(bleu1_scores),
            "BLEU-4": _mean(bleu4_scores),
            "METEOR": _mean(meteor_scores),
            "ROUGE-L": _mean(rouge_scores),
            "tIoU": _mean(tiou_scores),
        }

        if include_wts_metrics:
            results["CIDEr"] = compute_cider(hyp_answers, ref_answers)
            results["VQA_Acc"] = compute_vqa_accuracy(hyp_answers, ref_answers)

        for k, v in results.items():
            logger.info("%s: %.4f", k, v)

        return results
