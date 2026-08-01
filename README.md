# Conan-R1

**Conan-R1: Degradation-Aware Structured Reinforcement Learning for Traffic
Video Anomaly Understanding**

Conan-R1 uses `Qwen/Qwen2.5-VL-3B-Instruct`, LoRA, response-only supervised
fine-tuning (SFT), and group relative policy optimization (GRPO). It emits five
machine-parseable blocks:

```text
<TYPE>       degradation factors and severity                 <TYPE_END>
<INFLUENCE>  effect on observable-evidence reliability        <INFLUENCE_END>
<REASONING>  evidence-grounded explanatory reasoning          <REASONING_END>
<CONCLUSION> compact event-level judgment                     <CONCLUSION_END>
<ANSWER>     event_type plus interval in seconds and answer    <ANSWER_END>
```

The code uses “explanatory reasoning,” not formal causal inference. The four
reported rewards do not directly verify the semantic faithfulness of the
intermediate text, so the release does not claim causal identification.

## Release status

This source tree implements the corrected training and evaluation protocol.
It does not contain fabricated or unverified replacement numbers. Numerical
results require the version-matched Surv-VAU videos/manifests, model weights,
and CUDA hardware; those assets are not present in this local source package.
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
The issue-to-experiment mapping and execution order are frozen in
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md).
The rights holder must also resolve
[LICENSE_SELECTION_REQUIRED.md](LICENSE_SELECTION_REQUIRED.md) before calling
the repository open source.

Key corrections include:

- a stored rollout policy (`pi_old`) distinct from the frozen SFT reference
  policy (`pi_ref`), with reusable old token log-probabilities and active
  clipping diagnostics;
- four bounded, independently computable rewards: degradation agreement
  `r_d`, categorical event correctness `r_e`, temporal IoU `r_t`, and explicit
  reasoning-length control `r_l`;
- source-video-isolated 30%/70% training splits plus full-data,
  data-epoch-matched, and optimizer-step-matched SFT controls;
- full-minus-one reward ablations, a fixed-length control, structural
  ablation, the progressive `r_d` and `r_d+r_l` controls reported in the
  paper, and reward-weight sensitivity configurations;
- exact 25-frame timestamp-to-seconds prompts, strict interval parsing,
  standard SacreBLEU/METEOR/ROUGE-L/CIDEr/VQA implementations, event macro-F1,
  Recall@tIoU, and robustness retention/AUC;

## Fixed protocol

- backbone: `Qwen/Qwen2.5-VL-3B-Instruct`
- input: 25 uniformly sampled RGB frames resized to 224 × 224
- temporal unit: seconds, with all sampled timestamps included in the prompt
- output limit: 384 new tokens
- LoRA: rank 16, alpha 32, dropout 0.05
- SFT-30: 10 epochs, AdamW, learning rate 5e-5
- GRPO: 5 data epochs, group size 4, 2 update epochs, learning rate 1e-5
- GRPO sampling: temperature 0.9, top-p 0.95
- reward weights: `w_d=w_e=w_t=w_l=0.25`
- evaluation: greedy decoding
- fixed single-run seed: 42
- reference distributed hardware: 4 × 32 GB GPUs

This release intentionally does not run multiple seeds, report mean/standard
deviation, or calculate source-video bootstrap confidence intervals.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m nltk.downloader punkt wordnet omw-1.4
```

## Required data

Place the frozen release under `data/surv_vau/` as documented in
[data/README.md](data/README.md). The strict release gate is:

```bash
python scripts/validate_dataset.py \
  --data_dir data/surv_vau \
  --expect_sources 3688 \
  --expect_instances 27647 \
  --check_videos \
  --require_robustness_coverage

python scripts/report_training_budget.py \
  --data_dir data/surv_vau --world_size 4
```

The loader never silently replaces a missing training/evaluation video with
blank frames.

## Experiment configurations

Materialize immutable ablation and reward-sensitivity YAML files, then inspect
the complete dry-run command list:

```bash
python scripts/materialize_experiments.py \
  --matrix experiments/experiment_matrix.yaml \
  --output_dir configs/generated --overwrite

python scripts/run_experiment_suite.py --include_ablations
```

Only add `--execute` after the data validator passes. The core suite trains:

| Run | Additional data after SFT-30 | Purpose |
|---|---|---|
| SFT-30 | none | original 30% baseline |
| Continued-SFT-70 | remaining 70%, 5 epochs | data/data-epoch-matched RL control |
| Continued-SFT-70-Update-Matched | remaining 70%, 10 epochs | optimizer-step-matched RL control |
| SFT-100 | complete training partition | full-data SFT control |
| Conan-R1 | remaining 70%, 5 GRPO epochs | proposed RL stage |

Optimizer-step matching is not FLOP matching: GRPO additionally samples four
candidates and evaluates a reference policy. The generated training-budget
audit makes this distinction explicit.

To execute only the core suite on four GPUs:

```bash
python scripts/run_experiment_suite.py --execute
```

## Evaluation

Evaluate every internal and external system with the same raw-output scorer:

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/grpo_full \
  --model_name Conan-R1 \
  --data_dir data/surv_vau \
  --split test \
  --output results/conan_r1.json

python scripts/score_predictions.py external_predictions.jsonl \
  --data_dir data/surv_vau \
  --split test \
  --output results/external_method.json
```

By default, evaluation refuses to claim robustness unless the test manifest
contains 0%, 20%, 40%, and 80% levels and the `clean`, `synthetic_seen`,
`synthetic_unseen`, and `natural` domains. Use
`--allow_incomplete_robustness` only for a clearly labeled preliminary run.

External specialist comparisons and their fairness gate are recorded in
`experiments/baseline_protocol.yaml`. WTS uses the separate contract in
[data/WTS_PROTOCOL.md](data/WTS_PROTOCOL.md); `--wts` is a metric option, not a
dataset switch.

## Result handling

`results/paper_reported_pre_revision.json` contains only unverified values from
the earlier manuscript and is explicitly not a reproduction target. After a
real rerun, collect the observed single-run files:

```bash
python scripts/collect_results.py \
  results/base.json \
  results/sft30.json \
  results/continued_sft70.json \
  results/continued_sft70_update_matched.json \
  results/sft100.json \
  results/conan_r1.json
```

`scripts/verify_reproduction.py` accepts only a reference explicitly marked
`verified_release` and carrying a code revision plus annotation/split hashes.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m compileall -q .
pytest -q
```

The tests cover reward bounds, event aliases, temporal parsing, optional
structural blocks, group advantage normalization, non-negative sampled KL, and
a moved-policy case in which the probability ratio differs from one and
clipping becomes active.
