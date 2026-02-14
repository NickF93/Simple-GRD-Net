"""Training loop orchestration."""

from __future__ import annotations

import logging
from pathlib import Path

from torch.utils.data import DataLoader

from grdnet.backends.base import BackendStrategy
from grdnet.config.schema import ExperimentConfig
from grdnet.reporting.base import Reporter
from grdnet.training.checkpoints import save_checkpoint

LOGGER = logging.getLogger(__name__)


class TrainingEngine:
    """Epoch-based training loop with periodic validation and checkpointing."""

    def __init__(
        self,
        *,
        cfg: ExperimentConfig,
        backend: BackendStrategy,
        reporters: list[Reporter],
    ) -> None:
        self.cfg = cfg
        self.backend = backend
        self.reporters = reporters

    @staticmethod
    def _average(values: list[dict[str, float]]) -> dict[str, float]:
        if not values:
            return {}
        keys = sorted(values[0].keys())
        out: dict[str, float] = {}
        for key in keys:
            out[key] = float(sum(v[key] for v in values) / len(values))
        return out

    def _run_loader(self, loader: DataLoader, *, train: bool) -> dict[str, float]:
        all_stats: list[dict[str, float]] = []
        for step_idx, batch in enumerate(loader, start=1):
            output = self.backend.train_step(batch) if train else self.backend.eval_step(batch)
            all_stats.append(output.stats)
            if step_idx % self.cfg.training.log_interval == 0:
                LOGGER.info(
                    "step=%d mode=%s metrics=%s",
                    step_idx,
                    "train" if train else "val",
                    output.stats,
                )
        return self._average(all_stats)

    def train(self, train_loader: DataLoader, val_loader: DataLoader | None) -> None:
        checkpoint_dir = Path(self.cfg.training.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.cfg.training.epochs + 1):
            train_metrics = self._run_loader(train_loader, train=True)

            self.backend.schedulers.generator.step()
            self.backend.schedulers.discriminator.step()
            if self.backend.schedulers.segmentator is not None:
                self.backend.schedulers.segmentator.step()

            for reporter in self.reporters:
                reporter.log_epoch(epoch=epoch, split="train", metrics=train_metrics)

            if val_loader is not None and epoch % self.cfg.training.eval_interval == 0:
                val_metrics = self._run_loader(val_loader, train=False)
                for reporter in self.reporters:
                    reporter.log_epoch(epoch=epoch, split="val", metrics=val_metrics)

            checkpoint_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(self.backend, checkpoint_path, epoch=epoch)
