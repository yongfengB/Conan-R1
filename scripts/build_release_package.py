#!/usr/bin/env python3
"""Build a deterministic source-only ZIP and SHA256 file manifest."""
from __future__ import annotations

import argparse
import hashlib
import os
import zipfile
from pathlib import Path


EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".hypothesis",
    ".git",
    ".venv",
    "checkpoints",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".DS_Store"}
EXCLUDED_NAMES = {".DS_Store"}


def included(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if (
        path.suffix in EXCLUDED_SUFFIXES
        or path.name in EXCLUDED_NAMES
        or path.name == "MANIFEST.sha256"
    ):
        return False
    if relative.parts[:2] == ("data", "surv_vau"):
        return False
    if relative.parts and relative.parts[0] == "results":
        return path.name in {"README.md", "paper_results.json"}
    return True


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="../Conan-R1-paper-synchronized-code-2026-08-10.zip",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists.")
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and included(path, root)
    )
    manifest_lines = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        manifest_lines.append(
            f"{sha256_bytes(path.read_bytes())}  {relative}"
        )
    manifest_payload = ("\n".join(manifest_lines) + "\n").encode("utf-8")
    (root / "MANIFEST.sha256").write_bytes(manifest_payload)

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        archive_entries = [
            (path.relative_to(root).as_posix(), path.read_bytes())
            for path in files
        ]
        archive_entries.append(("MANIFEST.sha256", manifest_payload))
        for relative, payload in sorted(archive_entries):
            info = zipfile.ZipInfo(
                filename=f"Conan-R1/{relative}",
                date_time=(2026, 8, 10, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if relative.startswith("scripts/") else 0o644) << 16
            archive.writestr(info, payload)
    print(f"files={len(files) + 1}")
    print(f"zip={output}")
    print(f"zip_sha256={sha256_bytes(output.read_bytes())}")
    print(f"zip_bytes={os.path.getsize(output)}")


if __name__ == "__main__":
    main()
