"""Backend strategy abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from grdnet.config.schema import ExperimentConfig


@dataclass
class ModelBundle:
    """Container for model objects used by the selected backend."""

    generator: nn.Module
    discriminator: nn.Module
    segmentator: nn.Module | None


@dataclass
class OptimizerBundle:
    """Container for optimizers."""

    generator: Optimizer
    discriminator: Optimizer
    segmentator: Optimizer | None


@dataclass
class SchedulerBundle:
    """Container for schedulers."""

    generator: LRScheduler
    discriminator: LRScheduler
    segmentator: LRScheduler | None


@dataclass
class TrainBatchPreview:
    """Detached tensors used to render one train-batch preview artifact."""

    x: torch.Tensor
    x_noisy: torch.Tensor
    noise_mask: torch.Tensor
    x_rebuilt: torch.Tensor


@dataclass
class StepOutput:
    """Output payload for train/eval/inference step."""

    stats: dict[str, float]
    x_rebuilt: torch.Tensor
    patch_scores: torch.Tensor
    heatmap: torch.Tensor
    seg_map: torch.Tensor | None
    train_batch_preview: TrainBatchPreview | None = None


class BackendStrategy(ABC):
    """Strategy interface for backend-specific implementation details."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Runtime device used by the backend."""

    @abstractmethod
    def build_models(self) -> ModelBundle:
        """Create model objects."""

    @abstractmethod
    def build_optimizers(self, models: ModelBundle) -> OptimizerBundle:
        """Create optimizers for model bundle."""

    @abstractmethod
    def build_schedulers(self, optimizers: OptimizerBundle) -> SchedulerBundle:
        """Create scheduler bundle."""

    @abstractmethod
    def train_step(self, batch: dict[str, torch.Tensor | int | str]) -> StepOutput:
        """Execute one training step."""

    @abstractmethod
    def eval_step(self, batch: dict[str, torch.Tensor | int | str]) -> StepOutput:
        """Execute one evaluation step."""

    @abstractmethod
    def infer_step(self, batch: dict[str, torch.Tensor | int | str]) -> StepOutput:
        """Execute one inference step."""
