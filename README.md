# Conan-R1 core reference implementation

This repository provides the minimal reference implementation required to
audit the core method. It is not a self-contained reproduction package because
the full Surv-VAU video assets and some checkpoints are subject to distribution
restrictions.

The release contains the GRPO objective and trainer, four deterministic reward
functions, strict structured parser, unified evaluator, temporally coherent
degradation protocol, frozen SFT/GRPO/ablation configurations, annotation
schema, deterministic split code, a runnable synthetic demo, unit tests, and a
machine-readable transcription of the aggregate manuscript tables.

## Scope and evidence boundary

Included:

- `training/grpo_math.py` and `training/grpo_trainer.py`: `pi_old`, frozen
  `pi_ref`, clipped ratios, sampled reverse KL, group advantages, and updates;
- `training/rewards.py`: `r_d`, `r_e`, `r_t`, and `r_l`;
- `model/parser.py`, `evaluation/evaluator.py`, and `evaluation/metrics.py`:
  strict parsing and all reported metric implementations;
- `configs/`: full SFT, GRPO, progressive, full-minus-one, structure, and
  fixed-length configurations;
- `dataset/augmentation.py` and `configs/degradation_protocol.yaml`:
  trajectory-aware occlusion and video-level temporal degradation state;
- `data/annotation.schema.json` and `data/demo/`: schema plus a redistributable
  35-instance, 20-source executable dataset;
- `results/demo_raw_predictions.jsonl`: raw demo predictions used to test the
  builder-parser-tIoU interface;
- `results/paper_results.json`: aggregate values displayed in the manuscript.

Not included:

- original surveillance videos whose redistribution is restricted by privacy,
  collection authorization, or third-party licenses;
- large LoRA checkpoints; the exact training commands and frozen configs are
  provided instead;
- internal collection/cleaning tools and unrelated deployment, UI, monitoring,
  or human-study utilities.

`results/paper_results.json` is an aggregate table transcription. Paper-scale
raw predictions, checkpoint files, and the full Surv-VAU split manifest were
not present in the materials supplied for this core release, so the public
repository does not claim independent numerical reproduction from that JSON.
The included demo predictions are explicitly not evidence for manuscript-scale
performance.

## Environment

Python 3.10 or 3.11 is recommended. `requirements-lock.txt` is the frozen
reference environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m nltk.downloader punkt wordnet omw-1.4
```

## Run the complete demo audit

The committed demo contains 25-frame MP4 files, annotations, source-isolated
splits, a split manifest, and raw predictions. Regenerate it deterministically:

```bash
python scripts/create_demo_dataset.py
```

Validate data and split identities:

```bash
python scripts/validate_dataset.py \
  --data_dir data/demo \
  --expect_sources 20 \
  --expect_instances 35 \
  --check_videos \
  --require_robustness_coverage \
  --report results/demo_dataset_validation.json
```

Score its raw predictions with the same parser and evaluator:

```bash
python scripts/score_predictions.py results/demo_raw_predictions.jsonl \
  --model_name canonical-demo \
  --data_dir data/demo \
  --split test \
  --output results/demo_evaluation.json
```

The frozen demo identities are:

```text
annotations SHA256: b5280bc62a66e386233017762007d516dbf2a6fa6cd07b9fa224336072735e97
splits SHA256:      b097b305c682b8584a891c312d7b5eec0aa298d00600efc6d4963769ab4680c7
raw predictions:    580f09d43b98b4c7bc99d09e9b5f7e8600eb643044b15803dd1acee4707eb34a
```

## Output contract and duplicate-interval rule

```text
<TYPE>degradation factor and severity<TYPE_END>
<INFLUENCE>effect on observable evidence<INFLUENCE_END>
<REASONING>evidence-grounded trace<REASONING_END>
<CONCLUSION>compact event judgment<CONCLUSION_END>
<ANSWER>event_type: LABEL; interval: [start_sec, end_sec]<ANSWER_END>
```

`<ANSWER>` contains exactly one event label and one second-based interval.
Model-authored prose remains in `<REASONING>` and `<CONCLUSION>` and is never
appended to the answer. Missing, reversed, out-of-range, or multiple intervals
receive temporal reward zero. The unit tests exercise the complete
builder-parser-tIoU path, including an annotator response that itself contains
extra time expressions.

## Fixed core protocol

- backbone: `Qwen/Qwen2.5-VL-3B-Instruct`;
- 25 uniformly sampled RGB frames, resized to 224 by 224;
- LoRA rank 16, alpha 32, dropout 0.05;
- SFT: 10 epochs, AdamW, learning rate `5e-5`;
- GRPO: 5 data epochs, group size 4, 2 update epochs, learning rate `1e-5`;
- clipping epsilon `0.2`, KL coefficient `0.02`;
- reward weights `w_d=w_e=w_t=w_l=0.25`;
- generation: 384 tokens, temperature 0.9, top-p 0.95;
- evaluation: greedy decoding; seed 42.

The complete operator maxima, order, temporal state, `K` distribution, held-out
operators, and natural/synthetic domain contract are in
`configs/degradation_protocol.yaml`.

## Core configurations

```text
configs/sft_config.yaml
configs/grpo_config.yaml
configs/no_type_influence_sft_config.yaml
configs/generated/rd_only.yaml
configs/generated/rd_rl_only.yaml
configs/generated/without_rd.yaml
configs/generated/without_re.yaml
configs/generated/without_rt.yaml
configs/generated/without_rl.yaml
configs/generated/fixed_length_rl.yaml
configs/generated/without_type_influence.yaml
```

`experiments/experiment_matrix.yaml` records the commands and intervention
conditions. `configs/generated/SHA256SUMS.json` freezes every resolved ablation
configuration.

## Full-data access and identity

The schema and access requirements are documented in `data/README.md`. Requests
for authorized research access may be sent to `yongfengbu@chd.edu.cn`, copying
the corresponding authors `lhx@chd.edu.cn` and `kongli@xust.edu.cn`, with the
requester's affiliation, intended use, storage plan, and confirmation that no
identity or liability inference will be attempted. Access remains conditional
on the licenses and privacy terms of each source collection.

The full data split is source-video isolated and stratified by source dataset
and event type: 70/15/15 train/validation/test, followed by 30/70 SFT/GRPO
partitioning inside training, using seeds 42 and 43. A full-data release must
ship its own `annotations.jsonl`, `splits.json`, and `split_manifest.json`; the
validator rejects absent or mismatched hashes.

## Training, inference, and evaluation

```bash
python scripts/create_data_splits.py --data_dir data/surv_vau

python scripts/validate_dataset.py \
  --data_dir data/surv_vau \
  --expect_sources 3688 \
  --expect_instances 27647 \
  --check_videos \
  --require_robustness_coverage

torchrun --standalone --nproc_per_node=4 \
  scripts/train_sft.py --config configs/sft_config.yaml

torchrun --standalone --nproc_per_node=4 \
  scripts/train_grpo.py --config configs/grpo_config.yaml

python scripts/infer.py \
  --checkpoint checkpoints/grpo_full \
  --video /path/to/video.mp4 \
  --prompt "Identify the anomaly and temporal interval."

python scripts/evaluate.py \
  --checkpoint checkpoints/grpo_full \
  --model_name Conan-R1 \
  --data_dir data/surv_vau \
  --split test \
  --output results/conan_r1.json

python scripts/evaluate_interventions.py \
  --checkpoint checkpoints/grpo_full \
  --data_dir data/surv_vau \
  --split test \
  --output results/interventions.json
```

Every evaluation JSON records raw per-sample outputs and resolves the Git
revision, annotation hash, split hash, checkpoint hash, environment, command,
and decoding protocol at runtime.

## Tests and manuscript parity

```bash
python -m compileall -q .
pytest -q
python scripts/check_manuscript_sync.py ../../sections/06_experiments.tex
```

The parity check proves only that aggregate JSON and displayed LaTeX values are
identical. Empirical numerical verification additionally requires the
paper-scale raw predictions, checkpoint identity, and full-data manifest.
