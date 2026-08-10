"""Evaluator for structured Conan-R1 predictions."""
from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

from model.parser import (
    extract_event_type,
    extract_temporal_interval,
    parse_structured_output,
)
from .metrics import (
    compute_cider,
    compute_corpus_bleu,
    compute_event_metrics,
    compute_meteor,
    compute_rouge_l,
    compute_tiou,
    compute_tiou_recalls,
    compute_vqa_accuracy,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """Score only the answer block while retaining parsing diagnostics."""

    def evaluate(
        self,
        predictions: Sequence[str],
        references: Sequence[Dict],
        include_wts_metrics: bool = False,
    ) -> Tuple[Dict[str, float], List[Dict]]:
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have equal length.")

        answers, gt_answers, multi_references = [], [], []
        event_predictions, event_references = [], []
        tiou_scores, meteor_scores, rouge_scores = [], [], []
        details = []

        for prediction, reference in zip(predictions, references):
            parsed = parse_structured_output(prediction)
            answer = parsed.answer_block if parsed is not None else ""
            predicted_interval = extract_temporal_interval(answer)
            predicted_event = extract_event_type(answer)
            gt_answer = reference.get("answer_annotation", "")
            gt_interval = tuple(reference["gt_interval"])
            tiou = compute_tiou(
                predicted_interval,
                gt_interval,
                duration_sec=float(reference["duration_sec"]),
            )
            meteor = compute_meteor(answer, gt_answer)
            rouge_l = compute_rouge_l(answer, gt_answer)

            answers.append(answer)
            gt_answers.append(gt_answer)
            multi_references.append(
                reference.get("answer_references", [gt_answer])
            )
            event_predictions.append(predicted_event)
            event_references.append(reference["event_type"])
            tiou_scores.append(tiou)
            meteor_scores.append(meteor)
            rouge_scores.append(rouge_l)
            details.append(
                {
                    "video_id": reference.get("video_id", ""),
                    "source_video_id": reference.get("source_video_id", ""),
                    "parse_success": parsed is not None,
                    "predicted_answer": answer,
                    "ground_truth_answer": gt_answer,
                    "predicted_event_type": predicted_event,
                    "ground_truth_event_type": reference["event_type"],
                    "predicted_interval": predicted_interval,
                    "ground_truth_interval": list(gt_interval),
                    "tIoU": tiou,
                    "METEOR": meteor,
                    "ROUGE-L": rouge_l,
                    "degradation_level": float(
                        reference.get("degradation_level", 0.0)
                    ),
                    "degradation_domain": reference.get(
                        "degradation_domain", "synthetic_seen"
                    ),
                    "degradation_combination": reference.get(
                        "degradation_combination", "single_or_seen"
                    ),
                    "synthesis_applied": bool(
                        reference.get("synthesis_applied", False)
                    ),
                    "degradation_protocol": reference.get(
                        "degradation_protocol", "source_observation"
                    ),
                }
            )

        def mean(values: Sequence[float]) -> float:
            return float(sum(values) / max(1, len(values)))

        results = {
            "BLEU-1": compute_corpus_bleu(answers, gt_answers, max_order=1),
            "BLEU-4": compute_corpus_bleu(answers, gt_answers, max_order=4),
            "METEOR": mean(meteor_scores),
            "ROUGE-L": mean(rouge_scores),
            "tIoU": mean(tiou_scores),
            "Parse-Success": mean(
                [float(row["parse_success"]) for row in details]
            ),
        }
        results.update(compute_tiou_recalls(tiou_scores))
        results.update(
            compute_event_metrics(event_predictions, event_references)
        )
        if include_wts_metrics:
            results["CIDEr"] = compute_cider(answers, multi_references)
            results["VQA-Accuracy"] = compute_vqa_accuracy(
                answers, multi_references
            )
        for key, value in results.items():
            logger.info("%s: %.6f", key, value)
        return results, details
