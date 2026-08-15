#!/usr/bin/env python3
"""Verify that every paper table is bound to immutable executable evidence."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from scripts.collect_results import validate_result_artifact


REQUIRED_TABLES = tuple(f"Table {index}" for index in range(1, 8))
REPO_ROOT = Path(__file__).resolve().parents[1]


def _git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_paper_manifest(
    payload: dict,
    manifest_path: Path,
    *,
    check_local_tag: bool = True,
) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("Paper result manifest schema_version must be 1.")
    release = payload.get("release", {})
    tag = str(release.get("tag", ""))
    commit = str(release.get("commit", ""))
    if not tag or any(marker in tag.lower() for marker in ("placeholder", "draft")):
        raise ValueError("A non-placeholder immutable release tag is required.")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Release commit must be an exact 40-hex Git identity.")
    if check_local_tag and _git_output("rev-list", "-n", "1", tag) != commit:
        raise ValueError(f"Release tag {tag} does not resolve to {commit}.")
    tables = payload.get("tables", {})
    missing = [table for table in REQUIRED_TABLES if table not in tables]
    if missing:
        raise ValueError(f"Paper result manifest is missing tables: {missing}")
    root = manifest_path.parent
    for table in REQUIRED_TABLES:
        artifacts = tables[table]
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError(f"{table} must reference at least one result artifact.")
        for relative_path in artifacts:
            path = (root / relative_path).resolve()
            result = json.loads(path.read_text(encoding="utf-8"))
            validate_result_artifact(result, path)
            if result["provenance"]["code_revision"] != commit:
                raise ValueError(f"{path} was not produced by release commit {commit}.")
            if result["protocol"].get("table_id") != table:
                raise ValueError(f"{path} is not explicitly assigned to {table}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--skip-local-tag-check", action="store_true")
    args = parser.parse_args()
    path = Path(args.manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_paper_manifest(
        payload, path, check_local_tag=not args.skip_local_tag_check
    )
    print("paper result provenance: valid")


if __name__ == "__main__":
    main()
