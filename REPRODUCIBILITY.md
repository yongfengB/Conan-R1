# Reproducibility record

This revision separates code verification, experimental execution, and public
numerical reproduction.

## Implemented

- Fixed `Qwen/Qwen2.5-VL-3B-Instruct`, 25 frames, 224 × 224, seconds-based
  boundaries, greedy evaluation, and seed 42.
- Response-only LoRA SFT with optimizer/scheduler/trainer states and JSONL
  logs.
- Token-level GRPO with stored rollout log-probabilities, a separate frozen
  SFT reference policy, two update epochs, clipping/KL diagnostics, and saved
  optimizer/trainer state.
- Four normalized rewards (`r_d`, `r_e`, `r_t`, `r_l`) with validated weights
  summing to one.
- Independently annotated categorical event labels and manifest-controlled
  aliases; no same-family LLM judge is used for `r_e`.
- Source-video-level split creation and strict leakage/count/hash validation.
- SFT-30, data-epoch-matched Continued-SFT-70, optimizer-step-matched
  Continued-SFT-70, SFT-100, and Conan-R1 configurations.
- Full-minus-one rewards, fixed length, TYPE/INFLUENCE removal, and
  reward-weight sensitivity experiments.
- Standard lexical/task metrics, complete robustness-coverage validation,
  retention, normalized drop, robustness AUC, and raw per-sample outputs.
- Specialist-baseline fairness metadata and official-WTS provenance hooks.

## Not claimed in this source-only package

No new paper result is marked reproduced because this machine does not contain
the 27,647-instance Surv-VAU release, the 3,688 source videos, the required
model/adaptor weights, or CUDA hardware. The older manuscript values are kept
only in `results/paper_reported_pre_revision.json` with status
`unverified_pre_revision_values`.

Do not copy those values into revised tables. A table may be updated only from
an evaluation JSON produced by the matching checkpoint, frozen test manifest,
and current scorer.

## Numerical execution gate

Before training:

1. Complete `data/DATA_GOVERNANCE_CHECKLIST.md`.
2. Place `annotations.jsonl`, `splits.json`, `split_manifest.json`, and all
   videos under `data/surv_vau/`.
3. Include clean, seen-synthetic, unseen-synthetic, and natural test records at
   the four required degradation levels.
4. Run the strict dataset validator with video and robustness checks.
5. Generate `results/training_budget_audit.json`.
6. Materialize and hash all generated experiment YAML files.

After training:

1. Archive each resolved config, run metadata, training log, trainer state, and
   LoRA adapter.
2. Evaluate every system on the same frozen test list and retain raw outputs.
3. Run full-minus-one, fixed-length, structural, and weight-sensitivity
   experiments.
4. For WTS, record the official mapping hash and official scorer commit.
5. Collect only the observed single-run results.

## Public release gate

An exact public-reproducibility claim additionally requires:

- a stable code revision identifier;
- data access and license terms;
- annotation/split SHA256 hashes and the frozen test-video list;
- version-matched SFT and GRPO adapters or enough compute instructions to
  retrain them;
- the resolved environment and hardware record;
- raw prediction files for reported tables;
- completed privacy, authorization, anonymization, and bias documentation.

`scripts/verify_reproduction.py` rejects any reference file that is not marked
`verified_release` or lacks code/data provenance.
