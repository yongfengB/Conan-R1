# Reproducibility contract

The repository separates three reproducible artifacts:

1. **Method implementation** — fixed model, LoRA, SFT, token-level GRPO,
   bounded rewards, parser, metrics, and checkpoint state.
2. **Degradation generation** — a versioned operator protocol with explicit
   maxima, `K` distribution, composition rule, trajectory inputs, and
   video-level stochastic seed.
3. **Manuscript results** — `results/paper_results.json`, checked row by row
   against the LaTeX tables by `scripts/check_manuscript_sync.py`.

## Run identity

Every evaluation JSON records:

- the code revision;
- annotation and split SHA256 hashes;
- checkpoint SHA256 where applicable;
- the resolved model and decoding protocol;
- raw per-sample output and parsing status;
- degradation domain, level, and combination.

## Data identity

`scripts/validate_dataset.py` enforces:

- 3,688 immutable source-video identities and 27,647 instances;
- source-level 70/15/15 train/validation/test isolation;
- disjoint 30/70 SFT/GRPO training sources;
- valid timestamps and source-video metadata;
- paired 0/20/40/80 severity coverage;
- separate `clean`, `synthetic_seen`, `synthetic_unseen`, and `natural`
  domains;
- no synthetic-unseen or natural records in a training split;
- `synthesis_applied=false` for natural observations;
- matching annotation/split hashes.

## Numerical identity

The manuscript reference is deterministic and immutable within a release:

```bash
python scripts/check_manuscript_sync.py ../../sections/06_experiments.tex
```

Evaluation outputs can be frozen with `scripts/create_release_reference.py`
and compared with `scripts/verify_reproduction.py`. The comparison accepts
only matching code, annotation, and split identities.

## Claim boundary

The 20/40/80 table uses `synthetic_seen` operators. The evaluator produces
separate summaries for `synthetic_unseen` and `natural`; none of these domains
is silently relabeled as another. Text-overlap scores evaluate agreement with
the stored reference response and are not used alone as evidence of causal
faithfulness.
