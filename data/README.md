# Surv-VAU data contract and access procedure

The dataset root is:

```text
data/surv_vau/
├── annotations.jsonl
├── splits.json
├── split_manifest.json
└── videos/<video_id>.mp4
```

Every record includes the event/interval annotation, structured text fields,
source identity, degradation profile, severity, domain, combination,
`synthesis_applied`, and `degradation_protocol`.

The exact JSON field contract is `data/annotation.schema.json`. The committed
`data/demo/` directory is synthetic, redistributable, and executable; it is not
a sample of restricted surveillance footage and is not evidence for the paper's
aggregate performance. Its split manifest contains exact SHA256 values for the
annotations, split assignment, and demo raw predictions.

The four domains have distinct meanings:

- `clean`: no added operator;
- `synthetic_seen`: an operator from the training-time protocol;
- `synthetic_unseen`: a held-out operator or held-out combination, test only;
- `natural`: a naturally degraded source observation with no applied operator,
  test only.

For `natural`, set `synthesis_applied` to `false` and
`degradation_protocol` to `source_observation`. Synthetic-unseen and natural
records must never occur in `sft_train` or `rl_train`.

## Spatial annotations

Any source that uses `vehicle_mask` supplies `object_tracks`. Any source that
uses `interaction_area_mask` supplies either `interaction_regions` or at least
two overlapping event-relevant tracks. Boxes use normalized
`[x1, y1, x2, y2]` coordinates:

```json
{
  "track_id": "vehicle-17",
  "category": "vehicle",
  "event_relevant": true,
  "boxes": [
    {"frame_index": 0, "bbox_norm": [0.10, 0.40, 0.28, 0.66]},
    {"frame_index": 24, "bbox_norm": [0.58, 0.38, 0.80, 0.68]}
  ]
}
```

The builder interpolates boxes between annotated frames and raises an error if
a spatial operator has no admissible target.

## Validation

```bash
python scripts/validate_dataset.py \
  --data_dir data/surv_vau \
  --expect_sources 3688 \
  --expect_instances 27647 \
  --check_videos \
  --require_robustness_coverage
```

Source footage remains subject to its original license, consent, privacy, and
redistribution terms. This repository does not override those terms.

## Full-data access

Requests for authorized research access may be sent to
`yongfengbu@chd.edu.cn`, with copies to `lhx@chd.edu.cn` and
`kongli@xust.edu.cn`. Include affiliation, intended research use, storage and
access controls, requested source collections, and confirmation that the data
will not be used for identity inference or legal-liability decisions. Access
is conditional on each collection's license, privacy review, and redistribution
authority; third-party footage may need to be obtained from its original
provider.

A granted data release must include the matching `annotations.jsonl`,
`splits.json`, and `split_manifest.json`. The code refuses training or
evaluation when their hashes disagree. Large checkpoints are not bundled here;
they can be rebuilt with the frozen YAML files and commands in the root README.
