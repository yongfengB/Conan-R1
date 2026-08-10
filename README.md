# Conan-R1

**Degradation-Aware Structured Reinforcement Learning for Traffic Video
Anomaly Understanding**

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

The repository is synchronized with the manuscript in four concrete ways:

- `configs/degradation_protocol.yaml` is the source of truth for operators,
  maximum magnitudes, temporal models, composition order, and the distribution
  of the number of active factors `K`;
- `dataset/augmentation.py` implements object/trajectory-aware occlusion and
  video-level temporal state for weather, flare, noise, and motion blur;
- `results/paper_results.json` mirrors every numerical entry in the reported
  quantitative tables;
- `scripts/check_manuscript_sync.py` checks the JSON reference against the
  LaTeX table rows.

The structured trace is an inspectable diagnostic. The four rewards score
degradation agreement, event correctness, temporal overlap, and the explicit
length target; they do not by themselves prove causal faithfulness.

## Fixed paper protocol

- backbone: `Qwen/Qwen2.5-VL-3B-Instruct`
- input: 25 uniformly sampled RGB frames resized to 224 × 224
- temporal unit: seconds, with exact sampled timestamps in the prompt
- output limit: 384 new tokens
- LoRA: rank 16, alpha 32, dropout 0.05
- SFT: 10 epochs, AdamW, learning rate `5e-5`
- GRPO: 5 data epochs, group size 4, 2 update epochs, learning rate `1e-5`
- GRPO sampling: temperature 0.9, top-p 0.95
- reward weights: `w_d=w_e=w_t=w_l=0.25`
- evaluation: greedy decoding
- fixed single-run seed: 42
- reference distributed hardware: 4 × 32 GB GPUs

## Degradation protocol

The clean reference contains no added operator. Non-clean severity is one of
`0.2`, `0.4`, or `0.8`; it scales the published maximum of every active
operator. The number of active factors follows:

```text
P(K=1)=0.60, P(K=2)=0.30, P(K=3)=0.10
```

Single operators, cross-category pairs, and one-per-category triples use the
same `0.60/0.30/0.10` distribution and are sampled without replacement.

Spatial operators have no fixed-position fallback:

- `vehicle_mask` follows an event-relevant vehicle trajectory;
- `interaction_area_mask` follows an annotated interaction region or the
  stable closest pair of event-relevant tracks;
- missing required spatial metadata raises `SpatialAnnotationError`.

Temporal state is initialized once for the complete video/profile pair:

- rain/snow particles persist and advect across frames;
- lens flare follows a smooth origin/velocity path;
- sensor noise follows an AR(1) process with `rho=0.85`;
- motion-blur direction follows dominant tracked motion with smooth drift.

The exact maxima and composition order are in
[`configs/degradation_protocol.yaml`](configs/degradation_protocol.yaml).

## Robustness domains

Robustness is reported by domain and is never collapsed into one ambiguous
claim:

| Domain | Synthetic operator applied | Role |
|---|---:|---|
| `clean` | no | paired 0% reference |
| `synthetic_seen` | yes | training-time operators and paper severity table |
| `synthetic_unseen` | yes | held-out defocus/compression or held-out combination |
| `natural` | no | naturally degraded source observations, test only |

The paper's severity table is **synthetic-seen robustness**. It is not labeled
as real-world robustness. Natural and synthetic-unseen records are kept out of
both training subsets and summarized separately by the evaluator.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m nltk.downloader punkt wordnet omw-1.4
```

## Data contract

Prepare Surv-VAU under `data/surv_vau/` according to
[`data/README.md`](data/README.md). Source footage remains governed by its
original license or collection authorization and is not repackaged by this
source repository.

The validator checks source-level split isolation, exact counts, temporal
metadata, domain labels, synthesis provenance, held-out-domain leakage, and
paired severity coverage:

```bash
python scripts/validate_dataset.py \
  --data_dir data/surv_vau \
  --expect_sources 3688 \
  --expect_instances 27647 \
  --check_videos \
  --require_robustness_coverage
```

Object-aware synthesis consumes normalized frame-level trajectories:

```json
{
  "object_tracks": [{
    "track_id": "vehicle-17",
    "category": "vehicle",
    "event_relevant": true,
    "boxes": [
      {"frame_index": 0, "bbox_norm": [0.10, 0.40, 0.28, 0.66]},
      {"frame_index": 24, "bbox_norm": [0.58, 0.38, 0.80, 0.68]}
    ]
  }],
  "interaction_regions": [{
    "region_id": "contact-zone-1",
    "boxes": [
      {"frame_index": 12, "bbox_norm": [0.42, 0.46, 0.62, 0.70]}
    ]
  }]
}
```

Boxes are linearly interpolated only between annotated frames and use
normalized `[x1, y1, x2, y2]` coordinates.

## Training and evaluation

Build the dataset and run the paper configurations:

```bash
python scripts/build_dataset.py \
  --source_dir /path/to/authorized/videos \
  --annotation_file /path/to/source_annotations.json \
  --output_dir data/surv_vau

torchrun --standalone --nproc_per_node=4 \
  scripts/train_sft.py --config configs/sft_config.yaml

torchrun --standalone --nproc_per_node=4 \
  scripts/train_grpo.py --config configs/grpo_config.yaml

python scripts/evaluate.py \
  --checkpoint checkpoints/grpo_full \
  --model_name Conan-R1 \
  --data_dir data/surv_vau \
  --split test \
  --output results/conan_r1.json
```

External predictions use the same parser and scorer:

```bash
python scripts/score_predictions.py external_predictions.jsonl \
  --model_name METHOD \
  --data_dir data/surv_vau \
  --split test \
  --output results/external_method.json
```

## Manuscript-result parity

`results/paper_results.json` is the numerical reference for the tables in the
manuscript. In the paper workspace, check exact row-level parity with:

```bash
python scripts/check_manuscript_sync.py \
  ../../sections/06_experiments.tex
```

This check establishes paper-to-code consistency of the reported values. Raw
evaluation JSON files retain model, checkpoint, split, and scorer provenance
for empirical reproduction.

## Tests

```bash
python -m compileall -q .
pytest -q
```

The tests cover object-following masks, interaction-region targeting,
deterministic video-level state, temporal noise correlation, persistent
weather, `K` sampling, held-out operator sampling, reward bounds, temporal
parsing, GRPO math, metrics, and source-level split isolation.
