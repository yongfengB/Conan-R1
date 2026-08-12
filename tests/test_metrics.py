"""Metric and robustness protocol tests."""
import pytest

from evaluation.metrics import (
    compute_corpus_bleu,
    compute_event_metrics,
    compute_meteor,
    compute_rouge_l,
    compute_tiou_recalls,
    compute_vqa_accuracy,
    normalize_vqa_answer,
)
from evaluation.robustness import (
    summarize_robustness,
    validate_robustness_coverage,
)


def test_event_metrics():
    result = compute_event_metrics(
        ["collision", "fire", None],
        ["collision", "fire", "lane departure"],
    )
    assert result["Event-Accuracy"] == pytest.approx(2 / 3)
    assert 0.0 <= result["Event-Macro-F1"] <= 1.0


def test_event_metrics_honor_only_manifest_declared_aliases():
    result = compute_event_metrics(
        ["rear end crash", "rear end crash"],
        ["rear-end collision", "vehicle fire"],
        [["rear end crash"], []],
    )
    assert result["Event-Accuracy"] == pytest.approx(0.5)


def test_tiou_recalls():
    result = compute_tiou_recalls([0.2, 0.5, 0.8])
    assert result["Recall@tIoU=0.5"] == pytest.approx(2 / 3)
    assert result["Recall@tIoU=0.7"] == pytest.approx(1 / 3)


def test_vqa_normalization_and_single_reference_accuracy():
    assert normalize_vqa_answer("The red-car.") == "red car"
    assert compute_vqa_accuracy(["The red car"], ["red car"]) == 1.0


def test_standard_text_metrics_exact_match():
    pytest.importorskip("sacrebleu")
    pytest.importorskip("nltk")
    pytest.importorskip("rouge_score")
    hypotheses = ["rear end collision"]
    references = ["rear end collision"]
    assert compute_corpus_bleu(hypotheses, references, 1) == pytest.approx(1.0)
    assert compute_corpus_bleu(hypotheses, references, 4) == pytest.approx(1.0)
    assert compute_meteor(hypotheses[0], references[0]) == pytest.approx(1.0)
    assert compute_rouge_l(hypotheses[0], references[0]) == pytest.approx(1.0)


def test_robustness_requires_zero_percent_reference():
    rows = [
        {
            "source_video_id": "source-1",
            "degradation_level": 0.2,
            "degradation_domain": "synthetic_seen",
            "degradation_combination": "fog",
            "predicted_answer": "collision",
            "ground_truth_answer": "collision",
            "METEOR": 1.0,
            "ROUGE-L": 1.0,
            "tIoU": 1.0,
        }
    ]
    with pytest.raises(ValueError):
        summarize_robustness(rows)


def test_robustness_retention_and_auc():
    rows = []
    for level, score in ((0.0, 1.0), (0.2, 0.8), (0.4, 0.6), (0.8, 0.4)):
        rows.append(
            {
                "source_video_id": "source-1",
                "degradation_level": level,
                "degradation_domain": "clean" if level == 0.0 else "synthetic_seen",
                "degradation_combination": "none" if level == 0.0 else "fog",
                "predicted_answer": "collision",
                "ground_truth_answer": "collision",
                "METEOR": score,
                "ROUGE-L": score,
                "tIoU": score,
            }
        )
    report = summarize_robustness(
        rows, metrics=("METEOR", "ROUGE-L", "tIoU")
    )
    assert report["summary"]["METEOR"]["by_level"]["0.8"]["retention"] == pytest.approx(0.4)
    assert 0.0 < report["summary"]["METEOR"]["robustness_auc"] <= 1.0


def test_coverage_requires_natural_and_unseen_domains():
    rows = [
        {
            "source_video_id": "source-1",
            "degradation_level": level,
            "degradation_domain": domain,
            "synthesis_applied": domain.startswith("synthetic_"),
            "degradation_protocol": (
                "source_observation"
                if domain == "natural"
                else "surv-vau-degradation-v1"
            ),
        }
        for level, domain in (
            (0.0, "clean"),
            (0.2, "synthetic_seen"),
            (0.4, "synthetic_seen"),
            (0.8, "synthetic_seen"),
            (0.8, "synthetic_unseen"),
            (0.0, "natural"),
        )
    ]
    counts = validate_robustness_coverage(rows)
    assert counts["domain:natural"] == 1


def test_severity_curve_excludes_natural_and_synthetic_unseen_rows():
    rows = []
    for level, score in ((0.0, 1.0), (0.2, 0.8), (0.4, 0.6), (0.8, 0.4)):
        rows.append(
            {
                "source_video_id": "paired",
                "degradation_level": level,
                "degradation_domain": "clean" if level == 0.0 else "synthetic_seen",
                "degradation_combination": "none" if level == 0.0 else "fog",
                "predicted_answer": "collision",
                "ground_truth_answer": "collision",
                "METEOR": score,
                "ROUGE-L": score,
                "tIoU": score,
            }
        )
    rows.extend(
        [
            {
                **rows[0],
                "degradation_domain": "natural",
                "METEOR": 0.0,
            },
            {
                **rows[-1],
                "degradation_domain": "synthetic_unseen",
                "METEOR": 0.0,
            },
        ]
    )
    report = summarize_robustness(rows, metrics=("METEOR",))
    assert report["synthetic_seen_severity"]["0.0"]["METEOR"] == 1.0
    assert report["synthetic_seen_severity"]["0.8"]["METEOR"] == 0.4
    assert report["domains"]["natural"]["METEOR"] == 0.0
