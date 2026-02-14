"""Training loop orchestration."""

from __future__ import annotations

import logging
import math
from pathlib import Path

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from grdnet.backends.base import BackendStrategy, StepOutput
from grdnet.config.schema import ExperimentConfig
from grdnet.reporting.base import Reporter
from grdnet.reporting.train_batch_preview import TrainBatchPreviewWriter
from grdnet.training.checkpoints import save_checkpoint

LOGGER = logging.getLogger(__name__)


class TrainingEngine:
    """Epoch-based training loop with periodic validation and checkpointing."""

    _LOSS_POSTFIX_ABBREVIATIONS: dict[str, str] = {
        "adversarial": "adv",
        "contextual": "con",
        "discriminator": "disc",
        "encoder": "enc",
        "generator": "gen",
        "noise": "nse",
        "segmentator": "seg",
        "total_eval": "tot",
    }

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
        preview_cfg = self.cfg.reporting.train_batch_preview
        self._preview_writer: TrainBatchPreviewWriter | None = None
        if preview_cfg.enabled:
            self._preview_writer = TrainBatchPreviewWriter(
                output_dir=self.cfg.training.output_dir,
                subdir=preview_cfg.subdir,
                max_images=preview_cfg.max_images,
            )

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
    def _postfix_key(metric_key: str) -> str:
        if not metric_key.startswith("loss."):
            return metric_key

        suffix = metric_key.split(".", maxsplit=1)[1]
        suffix_display = TrainingEngine._LOSS_POSTFIX_ABBREVIATIONS.get(suffix, suffix)
        return f"l.{suffix_display}"

    @staticmethod
    def _postfix_metrics(metrics: dict[str, float]) -> dict[str, str]:
        ordered = sorted(metrics.items())
        out: dict[str, str] = {}
        for key, value in ordered:
            display_key = TrainingEngine._postfix_key(key)
            if display_key in out:
                display_key = key
            out[display_key] = f"{value:.5f}"
        return out

    @staticmethod
    def _optimizer_lr(optimizer: object) -> float | None:
        """Read LR from the first optimizer param-group when available."""
        param_groups = getattr(optimizer, "param_groups", None)
        if not isinstance(param_groups, list) or not param_groups:
            return None
        first = param_groups[0]
        if not isinstance(first, dict):
            return None
        raw = first.get("lr")
        if raw is None:
            return None
        lr = float(raw)
        if not math.isfinite(lr):
            return None
        return lr

    def _postfix_lrs(self) -> dict[str, str]:
        """Compact LR postfix values for progress-bar readability."""
        out: dict[str, str] = {}
        entries: list[tuple[str, object | None]] = [
            ("lr.g", self.backend.optimizers.generator),
            ("lr.d", self.backend.optimizers.discriminator),
            ("lr.s", self.backend.optimizers.segmentator),
        ]
        for key, optimizer in entries:
            if optimizer is None:
                continue
            lr = self._optimizer_lr(optimizer)
            if lr is None:
                continue
            out[key] = f"{lr:.5e}"
        return out

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
                self._maybe_write_train_batch_preview(
                    output=output,
                    train=train,
                    epoch=epoch,
                    step_idx=step_idx,
                )

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
                    postfix = self._postfix_metrics(averaged)
                    if train:
                        postfix.update(self._postfix_lrs())
                    progress.set_postfix(postfix)

        return self._average(all_stats)

    def _should_capture_train_preview(
        self,
        *,
        train: bool,
        epoch: int,
        step_idx: int,
    ) -> bool:
        preview_cfg = self.cfg.reporting.train_batch_preview
        if not train or not preview_cfg.enabled or self._preview_writer is None:
            return False
        if epoch < 1 or step_idx < 1:
            return False
        if step_idx != preview_cfg.step_index:
            return False
        return ((epoch - 1) % preview_cfg.every_n_epochs) == 0

    def _maybe_write_train_batch_preview(
        self,
        *,
        output: StepOutput,
        train: bool,
        epoch: int,
        step_idx: int,
    ) -> None:
        if not self._should_capture_train_preview(
            train=train,
            epoch=epoch,
            step_idx=step_idx,
        ):
            return
        if output.train_batch_preview is None:
            LOGGER.warning(
                "train_batch_preview_missing epoch=%d step=%d "
                "reason=backend_payload_absent",
                epoch,
                step_idx,
            )
            return
        if self._preview_writer is None:
            return
        path = self._preview_writer.write(
            epoch=epoch,
            step=step_idx,
            preview=output.train_batch_preview,
        )
        LOGGER.info(
            "train_batch_preview_saved epoch=%d step=%d path=%s",
            epoch,
            step_idx,
            path,
        )

    def _step_schedulers(self) -> None:
        self._maybe_step_scheduler(
            self.backend.schedulers.generator,
            optimizer=self.backend.optimizers.generator,
            name="generator",
        )
        self._maybe_step_scheduler(
            self.backend.schedulers.discriminator,
            optimizer=self.backend.optimizers.discriminator,
            name="discriminator",
        )
        if (
            self.backend.schedulers.segmentator is not None
            and self.backend.optimizers.segmentator is not None
        ):
            self._maybe_step_scheduler(
                self.backend.schedulers.segmentator,
                optimizer=self.backend.optimizers.segmentator,
                name="segmentator",
            )

    @staticmethod
    def _optimizer_stepped(optimizer: Optimizer) -> bool:
        """Check whether optimizer.step() was called in the current train step."""
        return bool(getattr(optimizer, "_opt_called", False))

    def _maybe_step_scheduler(
        self,
        scheduler: object,
        *,
        optimizer: Optimizer,
        name: str,
    ) -> None:
        if hasattr(scheduler, "optimizer") and not self._optimizer_stepped(optimizer):
            LOGGER.debug(
                "scheduler_step_skipped name=%s reason=optimizer_not_stepped",
                name,
            )
            return
        scheduler.step()

    def train(self, train_loader: DataLoader, val_loader: DataLoader | None) -> None:
        """Execute the full epoch loop with optional validation."""
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
