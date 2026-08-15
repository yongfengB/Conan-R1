# Conan-R1 core reference implementation

This repository provides the minimal reference implementation required to
audit the core method. It is not a self-contained reproduction package because
the full Surv-VAU video assets and some checkpoints are subject to distribution
restrictions.

The release contains the source-relative appearance--motion reliability
pathway, Qwen visual-token adapter, Stage-I/Stage-II objectives, task-masked
GRPO, four deterministic rewards, strict parser, unified evaluator,
temporally coherent degradation protocol, frozen experiment configurations,
annotation schema, deterministic split code, a runnable synthetic demo, and
unit tests.

## Scope and evidence boundary

Included:

- `model/reliability_pathway.py` and `model/qwen_adapter.py`: native-rate
  motion normalization, frozen/EMA teacher targets, occlusion adjustment,
  reliability-conditioned fusion, event pooling, temporal attention, and Qwen
  visual-token injection;
- `training/stage_objectives.py`: the Stage-I and Stage-II objectives retaining
  degradation, reliability, and consistency constraints;
- `training/grpo_math.py` and `training/grpo_trainer.py`: `pi_old`, frozen
  complete `pi_ref`, clipped ratios, sampled reverse KL, group advantages,
  task masks, active-weight renormalization, and full-policy updates;
- `training/rewards.py`: `r_d`, `r_e`, `r_t`, and `r_l`;
- `model/parser.py`, `evaluation/evaluator.py`, and `evaluation/metrics.py`:
  strict parsing and all reported metric implementations;
- `configs/`: method, full and LoRA-only SFT, GRPO, cumulative Stage-I,
  full-minus-one reward, matched-continuation, and appendix configurations;
- `dataset/augmentation.py` and `configs/degradation_protocol.yaml`:
  trajectory-aware occlusion and video-level temporal degradation state;
- `data/annotation.schema.json` and `data/demo/`: schema plus a redistributable
  32-instance, 20-source executable dataset;
- `results/demo_raw_predictions.jsonl`: raw demo predictions used to test the
  builder-parser-tIoU interface;

Not included:

- original surveillance videos whose redistribution is restricted by privacy,
  collection authorization, or third-party licenses;
- large LoRA checkpoints; the exact training commands and frozen configs are
  provided instead;
- internal collection/cleaning tools and unrelated deployment, UI, monitoring,
  or human-study utilities.

Paper-scale result JSON without matching raw predictions, complete checkpoints,
and the full Surv-VAU split manifest is deliberately excluded. The included demo
predictions are interface-audit evidence, not manuscript-scale evidence.

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

The committed demo contains 26-native-frame MP4 files, 25 anchor/adjacent-frame
pairs, annotations, source-isolated splits, a manifest, and raw predictions.

```bash
python scripts/create_demo_dataset.py
```

Validate data and split identities:

```bash
python scripts/validate_dataset.py \
  --data_dir data/demo \
  --expect_sources 20 \
  --expect_instances 32 \
  --check_videos \
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
annotations SHA256: 9fec9f5abd2c5b048a58683fae2974be27af2944a92927795531ef1ee48b71fb
splits SHA256:      2c01922699d4db9ed2ed600882c581c3dfd39d15015dd85d454e014dbb760771
raw predictions:    82691610c39947efd0075f174682c74c6147f880268e055d27f29c7e92cf2b54
```

## Output contract and duplicate-interval rule

```text
<TYPE>degradation factor and severity<TYPE_END>
<INFLUENCE>effect on observable evidence<INFLUENCE_END>
<REASONING>evidence-grounded trace<REASONING_END>
<CONCLUSION>compact event judgment<CONCLUSION_END>
<ANSWER>event_type: LABEL; interval: [start_sec, end_sec]<ANSWER_END>
```

`<ANSWER>` contains exactly the fields activated by the manifest task mask.
For a joint prompt this is one event label followed by one second-based interval.
Model-authored prose remains in `<REASONING>` and `<CONCLUSION>` and is never
appended to the answer. Missing, reversed, out-of-range, or multiple intervals
receive temporal reward zero. The unit tests exercise the complete
builder-parser-tIoU path, including an annotator response that itself contains
extra time expressions.

## Fixed core protocol

- backbone: `Qwen/Qwen2.5-VL-3B-Instruct`;
- Eq. (5) reliability target:
  `exp(-(1-cos(LN(F_d),LN(F_r)))/(2*tau_b))`, with
  `tau_appearance=tau_motion=0.25`; the loader rejects L2-labeled or
  formula-mismatched method configurations;
- 25 uniformly sampled RGB frames, resized to 224 by 224;
- native adjacent-frame motion divided by elapsed seconds and a fixed
  training-split `v_max`; scale fitting reuses the same resized-anchor and
  Farnebäck functions, streams one source at a time, and applies the recorded
  source-keyed deterministic pixel sample;
- diagnostic decoder slots are resolved only inside label-masked response
  tokens with fast-tokenizer character offsets; prompt markers are excluded,
  and invalid GRPO structures mask the undefined consistency term;
- frozen Qwen appearance encoder, frozen reference flow estimator, and EMA
  motion teacher with decay `0.999`;
- LoRA rank 16, alpha 32, dropout 0.05;
- SFT: 10 epochs, AdamW, learning rate `5e-5`;
- GRPO: 5 data epochs, group size 4, 2 update epochs, learning rate `1e-5`;
- clipping epsilon `0.2`, KL coefficient `0.02`;
- reward weights `w_d=w_e=w_t=w_l=0.25`;
- Eq. (12) compactness: effective reasoning length after deterministic repeated
  3--5-gram removal, a 64-token single-task or 96-token joint-task upper
  budget, and no minimum-length, severity-conditioned, or tolerance term;
- generation: 384 tokens, temperature 0.9, top-p 0.95;
- evaluation: greedy decoding; seed 42.

The complete operator maxima, order, temporal state, `K` distribution, held-out
operators, and natural/synthetic domain contract are in
`configs/degradation_protocol.yaml`.

## Core configurations

```text
configs/method_config.yaml
configs/sft_config.yaml
configs/structured_sft_config.yaml
configs/grpo_config.yaml
configs/no_type_influence_sft_config.yaml
configs/generated/without_rd.yaml
configs/generated/without_re.yaml
configs/generated/without_rt.yaml
configs/generated/without_rl.yaml
configs/generated/uniform_grpo.yaml
configs/generated/stage1_plus_motion.yaml
configs/generated/stage1_plus_reliability_supervision.yaml
configs/generated/stage1_plus_reliability_fusion.yaml
configs/generated/stage1_plus_event_pooling.yaml
configs/generated/stage1_plus_temporal_reliability.yaml
configs/generated/data_sft.yaml
configs/generated/update_sft.yaml
configs/generated/lora_grpo.yaml
configs/generated/pathway_grpo.yaml
configs/generated/conan_r1.yaml
```

`experiments/experiment_matrix.yaml` records the two-stage method, runnable SFT
controls, cumulative architecture, reward removals, matched continuations,
appendix controls, and five fixed-checkpoint reliability-field interventions.
Run `python scripts/materialize_experiments.py --overwrite` to regenerate all
complete YAML files and their SHA256 manifest. Configurations are not result evidence.

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

python scripts/estimate_motion_scale.py --data_dir data/surv_vau

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
  --robustness_scope synthetic \
  --table_id "Table 1" \
  --output results/conan_r1.json

python scripts/evaluate_interventions.py \
  --checkpoint checkpoints/grpo_full \
  --data_dir data/surv_vau \
  --split test \
  --table_id "Table 7" \
  --output results/interventions.json

python scripts/verify_paper_results.py results/paper_results_manifest.json
```

Every evaluation JSON records raw per-sample outputs and resolves the Git
revision, annotation hash, split hash, checkpoint hash, environment, command,
and decoding protocol at runtime.

Core checkpoint protocol version 6 binds the Eq. (5) metric, formula,
temperatures, the exact resized motion path, response-only diagnostic slots,
and the complete reliability-pathway config.

## Tests and manuscript parity

```bash
python -m compileall -q .
pytest -q
```

Paper-scale numerical verification additionally requires raw predictions,
checkpoint identity, and the full-data manifest; absent numerical artifacts are
not represented as result rows in this core repository.
