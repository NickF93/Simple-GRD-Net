"""Backend factory."""

from __future__ import annotations

from importlib import import_module
from typing import cast

from grdnet.backends.base import BackendStrategy
from grdnet.config.schema import ExperimentConfig
from grdnet.core.exceptions import BackendNotAvailableError

_BACKEND_SPECS: dict[str, tuple[str, str]] = {
    "pytorch": ("grdnet.backends.pytorch_backend", "PyTorchBackend"),
    "tensorflow_scaffold": (
        "grdnet.backends.tensorflow_backend",
        "TensorFlowScaffoldBackend",
    ),
}


def create_backend(cfg: ExperimentConfig) -> BackendStrategy:
    """Create backend implementation from config name."""
    spec = _BACKEND_SPECS.get(cfg.backend.name)
    if spec is None:
        raise BackendNotAvailableError(f"Unsupported backend: {cfg.backend.name}")

    module_path, backend_class_name = spec
    module = import_module(module_path)
    backend_class = cast(type[BackendStrategy], getattr(module, backend_class_name))
    return backend_class(cfg)
