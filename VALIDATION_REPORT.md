# Local validation report

Date: 2026-07-29 (Asia/Shanghai)

## Completed checks

- `python -m compileall -q .`: passed.
- Unit and property tests: **57 passed, 2 skipped** in the currently available
  local dependency set.
- The Hypothesis property suite was enabled from an isolated temporary
  dependency directory.
- Generated experiment configurations: **11/11 SHA256 hashes verified**.
- Experiment-suite dry run: command graph generated successfully.
- Python 3.9 compatibility: verified for the code-level tests.

The skipped tests depend on optional local packages that are absent from the
legacy Python environment, including the complete text-metric stack. A clean
training environment must install
`requirements-lock.txt`; the production evaluator deliberately raises an
error rather than substituting simplified metrics.

## Local environment

- Python: 3.9.7
- Platform: macOS 26.2, arm64
- Local PyTorch: 1.10.1
- CUDA available: no
- CUDA device count: 0

## Checks that cannot be completed on this machine

- Surv-VAU manifest/video validation: data assets absent.
- Qwen2.5-VL-3B-Instruct loading: model weights absent.
- SFT/GRPO execution: no CUDA GPU.
- New numerical table values: no matching data/checkpoints.
- Natural/unseen degradation robustness: evaluation assets absent.
- External specialist baselines: version-matched checkpoints/predictions
  absent.

No unavailable check is represented as passed, and no numerical result is
fabricated.
