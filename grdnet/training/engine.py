"""Training loop orchestration."""

from __future__ import annotations

import logging
import math
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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

    @staticmethod
    def _safe_total(loader: DataLoader) -> int | None:
        try:
            return len(loader)
        except TypeError:
            return None

    @staticmethod
    def _postfix_metrics(metrics: dict[str, float]) -> dict[str, str]:
        ordered = sorted(metrics.items())
        return {key: f"{value:.5f}" for key, value in ordered}

    def _run_loader(
        self,
        loader: DataLoader,
        *,
        train: bool,
        epoch: int,
    ) -> dict[str, float]:
        all_stats: list[dict[str, float]] = []
        running_totals: dict[str, float] = {}
        phase = "train" if train else "val"
        total_steps = self._safe_total(loader)

        with tqdm(
            loader,
            total=total_steps,
            desc=f"epoch={epoch} {phase}",
            dynamic_ncols=True,
            leave=False,
        ) as progress:
            for step_idx, batch in enumerate(progress, start=1):
                if train:
                    output = self.backend.train_step(batch)
                else:
                    output = self.backend.eval_step(batch)

                invalid_keys = [
                    key
                    for key, value in output.stats.items()
                    if not math.isfinite(value)
                ]
                if invalid_keys:
                    LOGGER.warning(
                        "non_finite_metrics_detected epoch=%d phase=%s step=%d keys=%s",
                        epoch,
                        phase,
                        step_idx,
                        invalid_keys,
                    )

                all_stats.append(output.stats)
                for key, value in output.stats.items():
                    running_totals[key] = running_totals.get(key, 0.0) + value

                if train and self.cfg.scheduler.step_unit == "step":
                    self._step_schedulers()

                if (
                    step_idx == 1
                    or step_idx % self.cfg.training.log_interval == 0
                    or (total_steps is not None and step_idx == total_steps)
                ):
                    averaged = {
                        key: running_totals[key] / step_idx
                        for key in sorted(running_totals)
                    }
                    progress.set_postfix(self._postfix_metrics(averaged))

        return self._average(all_stats)

    def _step_schedulers(self) -> None:
        self.backend.schedulers.generator.step()
        self.backend.schedulers.discriminator.step()
        if self.backend.schedulers.segmentator is not None:
            self.backend.schedulers.segmentator.step()

    def train(self, train_loader: DataLoader, val_loader: DataLoader | None) -> None:
        checkpoint_dir = Path(self.cfg.training.checkpoint_dir)
        LOGGER.info("preparing_checkpoint_directory path=%s", checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info(
            "starting_training epochs=%d scheduler_step_unit=%s",
            self.cfg.training.epochs,
            self.cfg.scheduler.step_unit,
        )

        for epoch in range(1, self.cfg.training.epochs + 1):
            LOGGER.info(
                "epoch_start epoch=%d/%d phase=train",
                epoch,
                self.cfg.training.epochs,
            )
            train_metrics = self._run_loader(train_loader, train=True, epoch=epoch)

            if self.cfg.scheduler.step_unit == "epoch":
                LOGGER.info("epoch_scheduler_step epoch=%d", epoch)
                self._step_schedulers()

            LOGGER.info("epoch_train_summary epoch=%d metrics=%s", epoch, train_metrics)
            for reporter in self.reporters:
                reporter.log_epoch(epoch=epoch, split="train", metrics=train_metrics)

            if val_loader is not None and epoch % self.cfg.training.eval_interval == 0:
                LOGGER.info(
                    "epoch_start epoch=%d/%d phase=val",
                    epoch,
                    self.cfg.training.epochs,
                )
                val_metrics = self._run_loader(val_loader, train=False, epoch=epoch)
                LOGGER.info("epoch_val_summary epoch=%d metrics=%s", epoch, val_metrics)
                for reporter in self.reporters:
                    reporter.log_epoch(epoch=epoch, split="val", metrics=val_metrics)

            checkpoint_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(self.backend, checkpoint_path, epoch=epoch)
            LOGGER.info("checkpoint_saved epoch=%d path=%s", epoch, checkpoint_path)

        LOGGER.info("training_completed epochs=%d", self.cfg.training.epochs)
