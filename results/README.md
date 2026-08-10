# Manuscript result reference

`paper_results.json` is the single numerical source synchronized with the
reported LaTeX tables. Each entry stores the table label, a row key that occurs
in the LaTeX source, and the displayed metrics.

Run the parity check from this repository inside the paper workspace:

```bash
python scripts/check_manuscript_sync.py ../../sections/06_experiments.tex
```

Evaluation runs should be stored as separate JSON files with raw per-sample
outputs and code/data/checkpoint provenance. The manuscript reference records
the values reported in the article; it is not overwritten by ad-hoc local
runs.
