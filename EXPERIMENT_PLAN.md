# Corrected single-run experiment plan

## Objective

Test whether degradation-aware structured GRPO improves traffic-video anomaly
event recognition and temporal localization beyond gains caused by additional
training data, update count, teacher-style lexical overlap, or prescribed
reasoning length.

## Fixed decisions

- One run per configuration, seed 42.
- No multi-seed mean/standard deviation.
- No source-video bootstrap confidence interval.
- Source-video-isolated train/validation/test assignment.
- Identical 25-frame, 224 × 224, seconds-based greedy evaluation.
- No table value is entered manually; all values come from saved evaluation
  JSON files with raw outputs and provenance.

## Required experiment groups

| Concern | Required runs/evidence | Primary decision |
|---|---|---|
| 30%/70% data confound | SFT-30, Continued-SFT-70, update-matched Continued-SFT-70, SFT-100, Conan-R1 | Conan-R1 must improve over matched continued-SFT controls, not only SFT-30 |
| Reward-to-task mismatch | four-reward Conan-R1 with event macro-F1 and Recall@tIoU | `r_e` must improve event correctness without reducing temporal performance |
| Additive ablation bias | full model minus each of `r_d`, `r_e`, `r_t`, `r_l` | report full-minus-one deltas |
| Length circularity | degradation-adaptive `r_l`, fixed 64-token `r_l`, and no-`r_l` | task gains must not depend only on matching prescribed length |
| Structural necessity | full five blocks vs no TYPE/INFLUENCE | show whether degradation blocks improve downstream event/tIoU |
| Reward sensitivity | equal, degradation-heavy, task-heavy, low-length weight sets | main conclusion should not rely on one fragile weight vector |
| Synthetic robustness | 0/20/40/80 plus seen and unseen factors/combinations | report absolute score, retention, normalized drop, AUC |
| Natural robustness | naturally degraded held-out test records | distinguish operator recognition from real observation degradation |
| Specialist comparisons | Cue-R1, GtS, VALU-compatible model, LRPO, RLER, TimeLens when executable | call a comparison fair only if the baseline protocol gate passes |
| WTS mismatch | frozen official mapping, standard metrics, separate official LLMScorer output | report mapping/scorer provenance |

## Execution order

1. Complete data governance and independent event/time annotations.
2. Freeze source splits and pass `scripts/validate_dataset.py`.
3. Generate the training-budget audit.
4. Materialize and review all generated YAML files.
5. Train the five core systems.
6. Evaluate the core systems and inspect raw parsing failures.
7. Train/evaluate reward, length, structure, and weight variants.
8. Run robustness evaluation.
9. Produce external baseline raw outputs and score them uniformly.
10. Collect observed result files and freeze a release reference.
11. Only then update manuscript tables, claims, equations, and captions.

## Failure rules

- Missing video, manifest field, split hash, or checkpoint is a hard error.
- Missing natural/unseen test domains cannot be relabeled as complete
  robustness.
- A baseline lacking matched training data/budget is reported as an unmatched
  transfer comparison.
- A teacher-referenced lexical score cannot be used alone to claim explanation
  correctness.
- Pre-revision values are never substituted for a failed or unavailable run.
