#!/usr/bin/env python3
"""Check exact row-level parity between paper_results.json and LaTeX tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "results" / "paper_results.json"


def _table_block(source: str, label: str) -> str:
    label_token = f"\\label{{{label}}}"
    label_index = source.find(label_token)
    if label_index < 0:
        raise ValueError(f"LaTeX table label not found: {label}")
    start = source.rfind("\\begin{table", 0, label_index)
    end = source.find("\\end{table}", label_index)
    if start < 0 or end < 0:
        raise ValueError(f"Could not isolate LaTeX table: {label}")
    return source[start : end + len("\\end{table}")]


def check_manuscript_sync(manuscript: Path, results_path: Path = DEFAULT_RESULTS) -> Dict[str, int]:
    source = manuscript.read_text(encoding="utf-8")
    reference = json.loads(results_path.read_text(encoding="utf-8"))
    row_count = 0
    value_count = 0
    for table in reference["tables"]:
        block = _table_block(source, table["latex_label"])
        lines = block.splitlines()
        for row in table["rows"]:
            matching_lines = [line for line in lines if row["latex_row_key"] in line]
            if len(matching_lines) != 1:
                raise ValueError(
                    f"{table['latex_label']} row key {row['latex_row_key']!r} "
                    f"matched {len(matching_lines)} lines"
                )
            latex_row = matching_lines[0]
            for metric, value in row["metrics"].items():
                precision = int(table["metric_precision"][metric])
                token = f"{float(value):.{precision}f}"
                if token not in latex_row:
                    raise ValueError(
                        f"{table['latex_label']} / {row['latex_row_key']} / {metric}: "
                        f"expected token {token}"
                    )
                value_count += 1
            row_count += 1
    return {"tables": len(reference["tables"]), "rows": row_count, "values": value_count}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manuscript", type=Path)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    summary = check_manuscript_sync(args.manuscript, args.results)
    print(json.dumps({"status": "in_sync", **summary}, indent=2))


if __name__ == "__main__":
    main()
