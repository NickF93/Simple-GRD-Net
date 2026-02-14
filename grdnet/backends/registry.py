"""Backend factory."""

from __future__ import annotations

from grdnet.backends.base import BackendStrategy
from grdnet.backends.pytorch_backend import PyTorchBackend
from grdnet.backends.tensorflow_backend import TensorFlowScaffoldBackend
from grdnet.config.schema import ExperimentConfig
from grdnet.core.exceptions import BackendNotAvailableError


def create_backend(cfg: ExperimentConfig) -> BackendStrategy:
    """Create backend implementation from config name."""
    if cfg.backend.name == "pytorch":
        return PyTorchBackend(cfg)
    if cfg.backend.name == "tensorflow_scaffold":
        return TensorFlowScaffoldBackend(cfg)
    raise BackendNotAvailableError(f"Unsupported backend: {cfg.backend.name}")
