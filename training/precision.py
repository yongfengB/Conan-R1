"""Mixed-precision safeguards shared by both training stages."""
from __future__ import annotations

import torch


def make_grad_scaler(device: str, dtype: torch.dtype):
    """Create a CUDA loss scaler only for FP16 optimization."""
    enabled = str(device).startswith("cuda") and dtype == torch.float16
    # torch.cuda.amp is available in every PyTorch release supported by this
    # reference package, including the locked 2.4 runtime.
    return torch.cuda.amp.GradScaler(enabled=enabled)


def require_finite(tensor: torch.Tensor, name: str) -> None:
    """Fail fast instead of committing a corrupted optimizer update."""
    if not bool(torch.isfinite(tensor.detach()).all().cpu()):
        raise FloatingPointError(f"Non-finite {name} encountered during training.")
