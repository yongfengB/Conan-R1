# Independent human-evaluation protocol

Use at least 300 distinct `source_video_id` values and three independent
raters per task. Raters see the video and two randomly ordered model outputs;
they do not see the model names, teacher annotation, or automatic metric
scores. The private A/B key must remain unavailable to raters until the
judgments are frozen.

Each rater independently supplies an event label and temporal interval, then
scores both candidates from 1 (incorrect) to 5 (fully correct) for event
correctness, temporal correctness, explanation correctness, evidence
groundedness, and sufficiency. Hallucination is a separate Boolean judgment.
The final field is an A/B/tie pairwise preference.

Create the blinded package:

```bash
python scripts/create_human_evaluation.py \
  --system Continued-SFT=results/continued_sft70.json \
  --system Conan-R1=results/conan_r1.json \
  --num_sources 300 \
  --output_dir human_evaluation
```

Store one completed JSON object per line in `ratings.jsonl`, following
`rating_schema.json`. Then compute model-level scores, Fleiss' kappa, and
pairwise temporal-boundary agreement:

```bash
python scripts/summarize_human_evaluation.py \
  --tasks human_evaluation/blinded_tasks.jsonl \
  --private_key human_evaluation/private_key.jsonl \
  --ratings human_evaluation/ratings.jsonl \
  --output human_evaluation/summary.json
```

The protocol intentionally reports the requested single-run descriptive
statistics without bootstrap confidence intervals.
