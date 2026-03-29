# Conan-R1

**Conan-R1: Structured Reasoning for Traffic Surveillance Video Anomaly Understanding with Observation-Difficulty Awareness**

A reinforcement learning-based framework that mitigates reasoning depth imbalance in multimodal large language models through structured observation-difficulty aware reasoning.

## Overview

Conan-R1 addresses the problem of *reasoning depth imbalance* in traffic surveillance video anomaly understanding. Built on Qwen2.5-VL-3B, it uses a two-stage training strategy (SFT + GRPO) to generate structured five-block outputs:

```
<TYPE>      observation difficulty factors & severity  </TYPE_END>
<INFLUENCE> how difficulty affects evidence reliability </INFLUENCE_END>
<REASONING> step-by-step causal reasoning chain        </REASONING_END>
<CONCLUSION> compact event-level judgment              </CONCLUSION_END>
<ANSWER>    final benchmark-compatible answer          </ANSWER_END>
```

## Installation

```bash
git clone https://github.com/your-org/conan-r1.git
cd conan-r1
pip install -r requirements.txt
```

Download NLTK data:
```python
import nltk
nltk.download('wordnet')
nltk.download('punkt')
```

## Dataset Preparation

### Using Surv-VAU

Place your source surveillance videos under `data/raw/` and run the five-stage pipeline:

```bash
python scripts/build_dataset.py \
    --source_dir data/raw \
    --output_dir data/surv_vau \
    --annotator_model Qwen/Qwen2.5-3B-Instruct
```

### Dataset Structure

```
data/surv_vau/
├── annotations.jsonl      # all structured samples
├── splits.json            # train/val/test split mapping
└── videos/                # degraded video clips
```

## Training

### Stage 1: Supervised Fine-Tuning (SFT)

```bash
python scripts/train_sft.py --config configs/sft_config.yaml
```

Override specific hyperparameters:
```bash
python scripts/train_sft.py --config configs/sft_config.yaml \
    --training.lr 3e-5 --training.epochs 5
```

### Stage 2: GRPO Reinforcement Learning

```bash
python scripts/train_grpo.py --config configs/grpo_config.yaml \
    --model.sft_checkpoint checkpoints/sft
```

## Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/grpo \
    --data_dir data/surv_vau \
    --split test
```

## Inference

```bash
python scripts/infer.py \
    --video path/to/video.mp4 \
    --checkpoint checkpoints/grpo \
    --prompt "Describe the traffic anomaly event and identify its temporal boundaries."
```

Save output to file:
```bash
python scripts/infer.py \
    --video path/to/video.mp4 \
    --checkpoint checkpoints/grpo \
    --output result.json
```

## Results on Surv-VAU

| Model | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | tIoU |
|-------|--------|--------|--------|---------|------|
| Qwen2.5-VL-3B (Base) | 0.2314 | 0.0527 | 0.2059 | 0.2198 | 0.3186 |
| + SFT | 0.3198 | 0.0786 | 0.3024 | 0.3308 | 0.4789 |
| **Conan-R1 (Ours)** | **0.3857** | **0.0965** | **0.3704** | **0.3872** | **0.6531** |

## Repository Structure

```
conan-r1/
├── dataset/           # Surv-VAU dataset builder & loader
├── model/             # ConanR1Model, LoRA, output parser
├── training/          # SFT & GRPO trainers, reward functions
├── evaluation/        # metrics (BLEU, METEOR, ROUGE-L, tIoU)
├── scripts/           # train, evaluate, infer CLI scripts
├── configs/           # YAML hyperparameter configs
├── tests/             # unit & property-based tests
└── requirements.txt
```


