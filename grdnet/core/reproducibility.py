"""Determinism and random seed controls."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from grdnet.core.determinism import (
    CUBLAS_WORKSPACE_ALLOWED,
    CUBLAS_WORKSPACE_ENV,
    is_valid_cublas_workspace,
)
from grdnet.core.exceptions import ConfigurationError


def validate_determinism_runtime(
    *,
    deterministic: bool,
    backend_name: str,
    backend_device: torch.device | None,
) -> None:
    """Validate CUDA determinism prerequisites for the selected runtime."""
    if not deterministic or backend_name != "pytorch":
        return
    if backend_device is None or backend_device.type != "cuda":
        return

    workspace = os.environ.get(CUBLAS_WORKSPACE_ENV)
    if is_valid_cublas_workspace(workspace):
        return

    allowed = ", ".join(sorted(CUBLAS_WORKSPACE_ALLOWED))
    raise ConfigurationError(
        "system.deterministic=true with CUDA requires "
        f"{CUBLAS_WORKSPACE_ENV} to be one of: {allowed}. "
        "Set it before launching Python so deterministic CUDA kernels are valid."
    )


def set_global_seed(seed: int, deterministic: bool) -> None:
    """Set all framework-level random seeds and deterministic flags."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Avoid CUDA runtime probing here; it can emit warnings in headless/CI setups.
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
