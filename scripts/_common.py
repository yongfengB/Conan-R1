"""Shared command-line utilities for reproducible Conan-R1 runs."""
from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: Optional[str]) -> str:
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def init_distributed(requested_device: Optional[str]) -> Tuple[str, int, int]:
    """Initialize torchrun/NCCL when WORLD_SIZE is greater than one."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size == 1:
        return resolve_device(requested_device), rank, world_size
    if requested_device and requested_device == "cpu":
        raise ValueError("The reference distributed run requires CUDA/NCCL.")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return f"cuda:{local_rank}", rank, world_size


def finish_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_component_hashes(checkpoint_root: Path) -> Dict[str, str]:
    """Hash every inference-defining file in a Conan-R1 checkpoint."""
    names = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "chat_template.json",
        "conan_core.pt",
        "conan_core_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    hashes = {
        path.name: digest
        for path in sorted(checkpoint_root.iterdir())
        if path.is_file()
        and path.name in names
        and (digest := sha256_file(path)) is not None
    } if checkpoint_root.is_dir() else {}
    return hashes


def checkpoint_identity(checkpoint_root: Path) -> Tuple[Optional[str], Dict[str, str]]:
    """Return a canonical identity for LoRA, core visual state, and processor."""
    components = checkpoint_component_hashes(checkpoint_root)
    if not components:
        return None, components
    payload = json.dumps(components, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), components


def load_core_protocol(checkpoint_root: Path) -> Dict[str, Any]:
    """Read the immutable, non-pickle protocol sidecar for a core checkpoint."""
    path = checkpoint_root / "conan_core_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Core checkpoint metadata not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 3:
        raise ValueError("Only Conan-R1 core checkpoint format version 3 is accepted.")
    if float(payload.get("motion_v_max", 0.0)) <= 0.0:
        raise ValueError("Core checkpoint has an invalid motion_v_max.")
    return payload


def code_revision() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_runtime_metadata(
    data_dir: Optional[str] = None,
    checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect environment and artifact provenance for a saved result."""
    metadata: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "code_revision": code_revision(),
        "command": sys.argv,
    }
    if data_dir:
        root = Path(data_dir)
        metadata["annotations_sha256"] = sha256_file(
            root / "annotations.jsonl"
        )
        metadata["splits_sha256"] = sha256_file(root / "splits.json")
        metadata["split_manifest_sha256"] = sha256_file(
            root / "split_manifest.json"
        )
    if checkpoint:
        checkpoint_root = Path(checkpoint)
        identity, components = checkpoint_identity(checkpoint_root)
        metadata["checkpoint_identity_sha256"] = identity
        metadata["checkpoint_files_sha256"] = components
    return metadata


def write_run_metadata(
    output_dir: str,
    config_path: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    with open(output / "resolved_config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    metadata = collect_runtime_metadata(
        data_dir=config.get("data", {}).get("data_dir"),
        checkpoint=output_dir,
    )
    metadata["config_sha256"] = sha256_file(Path(config_path))
    if extra:
        metadata.update(extra)
    with open(output / "run_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    try:
        frozen_environment = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        (output / "environment_freeze.txt").write_text(
            frozen_environment, encoding="utf-8"
        )
    except (OSError, subprocess.CalledProcessError):
        logging.getLogger(__name__).warning(
            "Could not record pip freeze for this run."
        )


def require_dataset(data_dir: str) -> None:
    root = Path(data_dir)
    missing = [
        str(root / name)
        for name in ("annotations.jsonl", "splits.json", "split_manifest.json")
        if not (root / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Surv-VAU is missing required files: "
            + ", ".join(missing)
            + ". Follow data/README.md before training."
        )
    split_manifest = json.loads(
        (root / "split_manifest.json").read_text(encoding="utf-8")
    )
    expected = {
        "annotations_sha256": sha256_file(root / "annotations.jsonl"),
        "splits_sha256": sha256_file(root / "splits.json"),
    }
    mismatched = [
        field
        for field, actual in expected.items()
        if split_manifest.get(field) != actual
    ]
    if mismatched:
        raise ValueError(
            "Dataset files do not match split_manifest.json: "
            + ", ".join(mismatched)
        )
