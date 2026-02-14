"""TensorFlow backend scaffold (placeholder-only in v1)."""

from __future__ import annotations

import torch

from grdnet.backends.base import (
    BackendStrategy,
    ModelBundle,
    OptimizerBundle,
    SchedulerBundle,
    StepOutput,
)
from grdnet.config.schema import ExperimentConfig


class TensorFlowScaffoldBackend(BackendStrategy):
    """Scaffold backend exposing future-compatible API contracts."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__(cfg)
        self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        """Return scaffold runtime device."""
        return self._device

    @staticmethod
    def _raise() -> None:
        raise NotImplementedError(
            "TensorFlow/Keras backend is scaffold-only in v1. "
            "Use backend.name='pytorch' for full execution."
        )

    def build_models(self) -> ModelBundle:
        """Scaffold method that raises deterministic not-implemented error."""
        self._raise()

    def build_optimizers(self, models: ModelBundle) -> OptimizerBundle:
        """Scaffold method that raises deterministic not-implemented error."""
        self._raise()

    def build_schedulers(self, optimizers: OptimizerBundle) -> SchedulerBundle:
        """Scaffold method that raises deterministic not-implemented error."""
        self._raise()

    def train_step(self, batch: dict[str, torch.Tensor | int | str]) -> StepOutput:
        """Scaffold method that raises deterministic not-implemented error."""
        self._raise()

    def eval_step(self, batch: dict[str, torch.Tensor | int | str]) -> StepOutput:
        """Scaffold method that raises deterministic not-implemented error."""
        self._raise()

    def infer_step(self, batch: dict[str, torch.Tensor | int | str]) -> StepOutput:
        """Scaffold method that raises deterministic not-implemented error."""
        self._raise()
