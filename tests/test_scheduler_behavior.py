from pathlib import Path

import pytest
import torch

from grdnet.backends.base import (
    OptimizerBundle,
    SchedulerBundle,
    StepOutput,
    TrainBatchPreview,
)
from grdnet.config.loader import load_experiment_config
from grdnet.training.engine import TrainingEngine
from grdnet.training.schedulers import GammaCosineAnnealingWarmRestarts


def test_gamma_scheduler_applies_decay_at_restart() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = GammaCosineAnnealingWarmRestarts(
        optimizer,
        first_restart_steps=2,
        restart_t_mult=1.0,
        restart_gamma=0.5,
    )

    initial = scheduler.get_last_lr()[0]
    optimizer.step()
    scheduler.step()  # step 0
    first = scheduler.get_last_lr()[0]
    optimizer.step()
    scheduler.step()  # step 1
    second = scheduler.get_last_lr()[0]
    optimizer.step()
    scheduler.step()  # step 2: restart with gamma
    restart = scheduler.get_last_lr()[0]

    assert initial == pytest.approx(1.0)
    assert first == pytest.approx(0.5)
    assert second == pytest.approx(0.5)
    assert restart == pytest.approx(0.25)


class _CountingScheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


class _OptimizerStub:
    def __init__(self, *, stepped: bool = True) -> None:
        self._opt_called = stepped


class _BackendStub:
    def __init__(self) -> None:
        generator = _CountingScheduler()
        discriminator = _CountingScheduler()
        self.schedulers = SchedulerBundle(
            generator=generator,
            discriminator=discriminator,
            segmentator=None,
        )
        self.optimizers = OptimizerBundle(
            generator=_OptimizerStub(),  # type: ignore[arg-type]
            discriminator=_OptimizerStub(),  # type: ignore[arg-type]
            segmentator=None,
        )

    def train_step(self, batch: dict[str, int]) -> StepOutput:
        _ = batch
        return StepOutput(
            stats={"loss": 1.0},
            x_rebuilt=torch.zeros((1, 1, 1, 1)),
            patch_scores=torch.zeros((1,)),
            heatmap=torch.zeros((1, 1, 1, 1)),
            seg_map=None,
        )

    def eval_step(self, batch: dict[str, int]) -> StepOutput:
        _ = batch
        return StepOutput(
            stats={"loss": 1.0},
            x_rebuilt=torch.zeros((1, 1, 1, 1)),
            patch_scores=torch.zeros((1,)),
            heatmap=torch.zeros((1, 1, 1, 1)),
            seg_map=None,
        )


def _engine_with_step_unit(
    step_unit: str,
    monkeypatch,
) -> tuple[TrainingEngine, _BackendStub]:
    cfg = load_experiment_config(Path("configs/profiles/deepindustrial_sn_2026.yaml"))
    cfg.scheduler.step_unit = step_unit
    cfg.training.epochs = 2
    cfg.training.log_interval = 1000
    cfg.training.checkpoint_dir = Path("/tmp/grdnet_test_checkpoints")
    cfg.training.output_dir = Path("/tmp/grdnet_test_reports")
    cfg.reporting.train_batch_preview.enabled = False

    backend = _BackendStub()
    monkeypatch.setattr(
        "grdnet.training.engine.save_checkpoint",
        lambda *args, **kwargs: None,
    )
    engine = TrainingEngine(cfg=cfg, backend=backend, reporters=[])
    return engine, backend


def test_training_engine_steps_scheduler_per_epoch(monkeypatch) -> None:
    engine, backend = _engine_with_step_unit("epoch", monkeypatch)
    train_loader = [{"x": 1}, {"x": 2}, {"x": 3}]
    engine.train(train_loader, None)
    assert backend.schedulers.generator.steps == 2
    assert backend.schedulers.discriminator.steps == 2


def test_training_engine_steps_scheduler_per_step(monkeypatch) -> None:
    engine, backend = _engine_with_step_unit("step", monkeypatch)
    train_loader = [{"x": 1}, {"x": 2}, {"x": 3}]
    engine.train(train_loader, None)
    assert backend.schedulers.generator.steps == 6
    assert backend.schedulers.discriminator.steps == 6


def test_postfix_metrics_uses_abbreviated_loss_keys() -> None:
    formatted = TrainingEngine._postfix_metrics(
        {
            "loss.generator": 1.23456,
            "loss.discriminator": 0.5,
            "loss.total_eval": 2.0,
            "score.mean": 0.125,
        }
    )
    assert "l.gen" in formatted
    assert "l.disc" in formatted
    assert "l.tot" in formatted
    assert "score.mean" in formatted
    assert formatted["l.gen"] == "1.23456"


def test_postfix_lrs_uses_compact_keys_and_scientific_format(monkeypatch) -> None:
    engine, backend = _engine_with_step_unit("step", monkeypatch)
    generator_param = torch.nn.Parameter(torch.tensor([1.0]))
    discriminator_param = torch.nn.Parameter(torch.tensor([2.0]))
    backend.optimizers = OptimizerBundle(
        generator=torch.optim.SGD([generator_param], lr=1e-2),
        discriminator=torch.optim.SGD([discriminator_param], lr=2e-2),
        segmentator=None,
    )

    postfix_lrs = engine._postfix_lrs()
    assert postfix_lrs == {
        "lr.g": "1.00000e-02",
        "lr.d": "2.00000e-02",
    }


def test_training_engine_skips_scheduler_when_optimizer_not_stepped(
    monkeypatch,
) -> None:
    cfg = load_experiment_config(Path("configs/profiles/deepindustrial_sn_2026.yaml"))
    cfg.scheduler.step_unit = "step"
    cfg.training.epochs = 1
    cfg.training.log_interval = 1000
    cfg.training.checkpoint_dir = Path("/tmp/grdnet_test_checkpoints")
    cfg.training.output_dir = Path("/tmp/grdnet_test_reports")
    cfg.reporting.train_batch_preview.enabled = False

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = GammaCosineAnnealingWarmRestarts(
        optimizer,
        first_restart_steps=2,
        restart_t_mult=1.0,
        restart_gamma=0.5,
    )

    class _BackendNoOptStep:
        def __init__(self) -> None:
            self.optimizers = OptimizerBundle(
                generator=optimizer,
                discriminator=optimizer,
                segmentator=None,
            )
            self.schedulers = SchedulerBundle(
                generator=scheduler,
                discriminator=scheduler,
                segmentator=None,
            )

        def train_step(self, batch: dict[str, int]) -> StepOutput:
            _ = batch
            optimizer._opt_called = False  # type: ignore[attr-defined]
            return StepOutput(
                stats={"loss": 1.0},
                x_rebuilt=torch.zeros((1, 1, 1, 1)),
                patch_scores=torch.zeros((1,)),
                heatmap=torch.zeros((1, 1, 1, 1)),
                seg_map=None,
            )

        def eval_step(self, batch: dict[str, int]) -> StepOutput:
            _ = batch
            return StepOutput(
                stats={"loss": 1.0},
                x_rebuilt=torch.zeros((1, 1, 1, 1)),
                patch_scores=torch.zeros((1,)),
                heatmap=torch.zeros((1, 1, 1, 1)),
                seg_map=None,
            )

    backend = _BackendNoOptStep()
    monkeypatch.setattr(
        "grdnet.training.engine.save_checkpoint",
        lambda *args, **kwargs: None,
    )
    engine = TrainingEngine(cfg=cfg, backend=backend, reporters=[])

    train_loader = [{"x": 1}, {"x": 2}]
    engine.train(train_loader, None)

    # LRScheduler initializes with last_epoch=0 in current torch.
    assert scheduler.last_epoch == 0


def test_training_engine_writes_train_preview_on_first_step(monkeypatch) -> None:
    cfg = load_experiment_config(Path("configs/profiles/deepindustrial_sn_2026.yaml"))
    cfg.scheduler.step_unit = "epoch"
    cfg.training.epochs = 2
    cfg.training.log_interval = 1000
    cfg.training.checkpoint_dir = Path("/tmp/grdnet_test_checkpoints")
    cfg.training.output_dir = Path("/tmp/grdnet_test_reports")
    cfg.reporting.train_batch_preview.enabled = True
    cfg.reporting.train_batch_preview.every_n_epochs = 1
    cfg.reporting.train_batch_preview.step_index = 1

    class _BackendWithPreview:
        def __init__(self) -> None:
            scheduler = _CountingScheduler()
            self.optimizers = OptimizerBundle(
                generator=_OptimizerStub(),  # type: ignore[arg-type]
                discriminator=_OptimizerStub(),  # type: ignore[arg-type]
                segmentator=None,
            )
            self.schedulers = SchedulerBundle(
                generator=scheduler,
                discriminator=scheduler,
                segmentator=None,
            )

        def train_step(self, batch: dict[str, int]) -> StepOutput:
            _ = batch
            preview = TrainBatchPreview(
                x=torch.zeros((2, 1, 8, 8)),
                x_noisy=torch.zeros((2, 1, 8, 8)),
                noise_mask=torch.zeros((2, 1, 8, 8)),
                x_rebuilt=torch.zeros((2, 1, 8, 8)),
            )
            return StepOutput(
                stats={"loss": 1.0},
                x_rebuilt=torch.zeros((1, 1, 1, 1)),
                patch_scores=torch.zeros((1,)),
                heatmap=torch.zeros((1, 1, 1, 1)),
                seg_map=None,
                train_batch_preview=preview,
            )

        def eval_step(self, batch: dict[str, int]) -> StepOutput:
            _ = batch
            return StepOutput(
                stats={"loss": 1.0},
                x_rebuilt=torch.zeros((1, 1, 1, 1)),
                patch_scores=torch.zeros((1,)),
                heatmap=torch.zeros((1, 1, 1, 1)),
                seg_map=None,
            )

    class _PreviewWriterStub:
        def __init__(self, **_kwargs) -> None:
            self.calls: list[tuple[int, int]] = []

        def write(self, *, epoch: int, step: int, preview: TrainBatchPreview) -> Path:
            _ = preview
            self.calls.append((epoch, step))
            return Path(f"/tmp/epoch_{epoch:04d}_step_{step:04d}.png")

    writer_instances: list[_PreviewWriterStub] = []

    def _writer_factory(**kwargs) -> _PreviewWriterStub:
        _ = kwargs
        writer = _PreviewWriterStub()
        writer_instances.append(writer)
        return writer

    monkeypatch.setattr(
        "grdnet.training.engine.TrainBatchPreviewWriter",
        _writer_factory,
    )
    monkeypatch.setattr(
        "grdnet.training.engine.save_checkpoint",
        lambda *args, **kwargs: None,
    )

    engine = TrainingEngine(cfg=cfg, backend=_BackendWithPreview(), reporters=[])
    train_loader = [{"x": 1}, {"x": 2}]
    engine.train(train_loader, None)

    assert len(writer_instances) == 1
    assert writer_instances[0].calls == [(1, 1), (2, 1)]
