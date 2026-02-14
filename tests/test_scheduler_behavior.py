from pathlib import Path

import pytest
import torch

from grdnet.backends.base import SchedulerBundle, StepOutput
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

    scheduler.step()  # step 0
    first = scheduler.get_last_lr()[0]
    scheduler.step()  # step 1
    second = scheduler.get_last_lr()[0]
    scheduler.step()  # step 2: restart with gamma
    restart = scheduler.get_last_lr()[0]

    assert first == pytest.approx(1.0)
    assert second == pytest.approx(0.5)
    assert restart == pytest.approx(0.5)


class _CountingScheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


class _BackendStub:
    def __init__(self) -> None:
        generator = _CountingScheduler()
        discriminator = _CountingScheduler()
        self.schedulers = SchedulerBundle(
            generator=generator,
            discriminator=discriminator,
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
