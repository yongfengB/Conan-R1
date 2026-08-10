# Reproducibility and provenance contract

This is a core reference implementation, not a self-contained full-scale
reproduction package.

Three artifact classes are kept separate:

1. **Core method audit**: source, configs, schema, parser, evaluator, tests, and
   degradation protocol are included.
2. **Executable demo audit**: synthetic videos, annotations, split manifest,
   raw predictions, and exact SHA256 identities are included.
3. **Manuscript numerical audit**: aggregate table JSON is included, while the
   paper-scale raw predictions, checkpoint files, and full-data split manifest
   are external to this release.

Every new training or evaluation run must record:

- `git rev-parse HEAD` for the code revision;
- `annotations.jsonl`, `splits.json`, and `split_manifest.json` SHA256 values;
- LoRA checkpoint filename and SHA256;
- resolved YAML and its SHA256;
- Python, PyTorch, CUDA, GPU, command, seed, and decoding settings;
- raw output, parse status, event label, interval, and per-sample metrics.

The command-line runners implement this contract through
`scripts/_common.py`. A numerical table is independently auditable only when
its aggregate JSON can be traced to the corresponding raw outputs and all
three code/data/checkpoint identities.

The manuscript parity command

```bash
python scripts/check_manuscript_sync.py ../../sections/06_experiments.tex
```

checks transcription parity, not empirical provenance.
