# Reproducibility and provenance contract

This is a core reference implementation, not a self-contained full-scale
reproduction package.

Three artifact classes are kept separate:

1. **Core method audit**: source, configs, schema, parser, evaluator, tests, and
   degradation protocol are included.
2. **Executable demo audit**: synthetic videos, annotations, split manifest,
   raw predictions, and exact SHA256 identities are included.
3. **Paper-scale numerical audit**: result rows are accepted only together with
   raw predictions, complete checkpoint identity, and the full-data split
   manifest. No paper-scale numerical rows are included in this core release.

Every new training or evaluation run must record:

- `git rev-parse HEAD` for the code revision;
- `annotations.jsonl`, `splits.json`, and `split_manifest.json` SHA256 values;
- every inference-defining checkpoint component SHA256 plus a canonical
  checkpoint identity;
- resolved YAML and its SHA256;
- Python, PyTorch, CUDA, GPU, command, seed, and decoding settings;
- raw output, parse status, event label, interval, and per-sample metrics.

The command-line runners implement this contract through
`scripts/_common.py`. A numerical table is independently auditable only when
its aggregate JSON can be traced to the corresponding raw outputs and all
three code/data/checkpoint identities.

Core checkpoint protocol version 5 additionally binds the exact Eq. (5)
layer-normalized cosine formula and reliability configuration. Training refuses
a metric/formula mismatch. The reward and data loaders likewise reject the
superseded sample-level target-length/tolerance interface for Eq. (12).

The evaluator and result collector operate only on raw outputs. A copied table
value without the required code, data, and checkpoint identities is not a
release artifact.
