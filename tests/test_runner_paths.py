from pathlib import Path
from types import SimpleNamespace

import pytest

from grdnet.pipeline import runner


class _DataModuleStub:
    def __init__(self, *, has_test: bool = True, has_cal: bool = True) -> None:
        self._has_test = has_test
        self._has_cal = has_cal

    def train_loader(self):
        return [{"x": 1}]

    def val_loader(self):
        return [{"x": 1}]

    def test_loader(self):
        return [{"x": 1}] if self._has_test else None

    def calibration_loader(self):
        return [{"x": 1}] if self._has_cal else None


def _cfg(threshold):
    return SimpleNamespace(
        inference=SimpleNamespace(anomaly_threshold=threshold),
    )


def test_setup_wires_dependencies(monkeypatch) -> None:
    cfg = SimpleNamespace(
        system=SimpleNamespace(log_level="INFO", seed=1, deterministic=True),
        data=SimpleNamespace(),
    )
    backend_obj = object()
    datamodule_obj = object()
    csv_reporter_obj = object()
    console_reporter_obj = object()

    monkeypatch.setattr(
        "grdnet.pipeline.runner.load_experiment_config",
        lambda path: cfg,
    )
    monkeypatch.setattr(
        "grdnet.pipeline.runner.configure_logging",
        lambda level: None,
    )
    monkeypatch.setattr(
        "grdnet.pipeline.runner.set_global_seed",
        lambda seed, deterministic: None,
    )
    monkeypatch.setattr(
        "grdnet.pipeline.runner.create_backend",
        lambda in_cfg: backend_obj,
    )
    monkeypatch.setattr(
        "grdnet.pipeline.runner.DataModule",
        lambda data_cfg: datamodule_obj,
    )
    monkeypatch.setattr(
        "grdnet.pipeline.runner.ConsoleReporter",
        lambda: console_reporter_obj,
    )
    monkeypatch.setattr(
        "grdnet.pipeline.runner.CsvReporter",
        lambda in_cfg: csv_reporter_obj,
    )

    out_cfg, out_backend, out_datamodule, reporters = runner._setup("cfg.yaml")
    assert out_cfg is cfg
    assert out_backend is backend_obj
    assert out_datamodule is datamodule_obj
    assert reporters == [console_reporter_obj, csv_reporter_obj]


def test_maybe_load_checkpoint_no_checkpoint_logs_warning(monkeypatch) -> None:
    messages: list[str] = []
    monkeypatch.setattr(
        "grdnet.pipeline.runner.LOGGER.warning",
        lambda message: messages.append(message),
    )
    runner._maybe_load_checkpoint(backend=object(), checkpoint=None)
    assert messages


def test_maybe_load_checkpoint_loads_epoch(monkeypatch, tmp_path: Path) -> None:
    messages: list[str] = []
    monkeypatch.setattr(
        "grdnet.pipeline.runner.load_checkpoint",
        lambda backend, path: 7,
    )
    monkeypatch.setattr(
        "grdnet.pipeline.runner.LOGGER.info",
        lambda message, epoch: messages.append(f"{message}:{epoch}"),
    )
    runner._maybe_load_checkpoint(
        backend=object(),
        checkpoint=str(tmp_path / "ckpt.pt"),
    )
    assert messages


def test_run_train_invokes_training_engine(monkeypatch) -> None:
    calls = {"train": 0}

    class _TrainingEngineStub:
        def __init__(self, *, cfg, backend, reporters) -> None:
            _ = cfg, backend, reporters

        def train(self, train_loader, val_loader) -> None:
            _ = train_loader, val_loader
            calls["train"] += 1

    monkeypatch.setattr(
        "grdnet.pipeline.runner._setup",
        lambda config_path: (_cfg(None), object(), _DataModuleStub(), []),
    )
    monkeypatch.setattr("grdnet.pipeline.runner.TrainingEngine", _TrainingEngineStub)
    assert runner.run_train("cfg.yaml") == 0
    assert calls["train"] == 1


def test_run_calibrate_happy_path(monkeypatch) -> None:
    class _InferenceEngineStub:
        def __init__(self, *, cfg, backend, reporters) -> None:
            _ = cfg, backend, reporters

        def calibrate(self, loader) -> float:
            _ = loader
            return 0.42

    monkeypatch.setattr(
        "grdnet.pipeline.runner._setup",
        lambda config_path: (_cfg(None), object(), _DataModuleStub(has_cal=True), []),
    )
    monkeypatch.setattr(
        "grdnet.pipeline.runner._maybe_load_checkpoint",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("grdnet.pipeline.runner.InferenceEngine", _InferenceEngineStub)
    assert runner.run_calibrate("cfg.yaml", checkpoint=None) == 0


def test_run_calibrate_requires_calibration_loader(monkeypatch) -> None:
    monkeypatch.setattr(
        "grdnet.pipeline.runner._setup",
        lambda config_path: (_cfg(None), object(), _DataModuleStub(has_cal=False), []),
    )
    monkeypatch.setattr(
        "grdnet.pipeline.runner._maybe_load_checkpoint",
        lambda *args, **kwargs: None,
    )
    with pytest.raises(ValueError, match="data.calibration_dir"):
        runner.run_calibrate("cfg.yaml", checkpoint=None)


def test_run_evaluate_with_explicit_threshold(monkeypatch) -> None:
    class _InferenceEngineStub:
        def __init__(self, *, cfg, backend, reporters) -> None:
            _ = cfg, backend, reporters

        def evaluate(self, loader, threshold):
            _ = loader, threshold
            return {"accuracy": 1.0}

    monkeypatch.setattr(
        "grdnet.pipeline.runner._setup",
        lambda config_path: (_cfg(0.1), object(), _DataModuleStub(), []),
    )
    monkeypatch.setattr(
        "grdnet.pipeline.runner._maybe_load_checkpoint",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("grdnet.pipeline.runner.InferenceEngine", _InferenceEngineStub)
    assert runner.run_evaluate("cfg.yaml", checkpoint=None) == 0


def test_run_evaluate_requires_threshold_or_calibration_loader(monkeypatch) -> None:
    class _InferenceEngineStub:
        def __init__(self, *, cfg, backend, reporters) -> None:
            _ = cfg, backend, reporters

    monkeypatch.setattr(
        "grdnet.pipeline.runner._setup",
        lambda config_path: (_cfg(None), object(), _DataModuleStub(has_cal=False), []),
    )
    monkeypatch.setattr(
        "grdnet.pipeline.runner._maybe_load_checkpoint",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("grdnet.pipeline.runner.InferenceEngine", _InferenceEngineStub)
    with pytest.raises(ValueError, match="No threshold available"):
        runner.run_evaluate("cfg.yaml", checkpoint=None)


def test_run_infer_uses_calibration_when_threshold_missing(monkeypatch) -> None:
    calls = {"calibrate": 0, "infer": 0}

    class _InferenceEngineStub:
        def __init__(self, *, cfg, backend, reporters) -> None:
            _ = cfg, backend, reporters

        def calibrate(self, loader) -> float:
            _ = loader
            calls["calibrate"] += 1
            return 0.3

        def infer(self, loader, threshold):
            _ = loader, threshold
            calls["infer"] += 1
            return [{"path": "a.png"}]

    monkeypatch.setattr(
        "grdnet.pipeline.runner._setup",
        lambda config_path: (_cfg(None), object(), _DataModuleStub(), []),
    )
    monkeypatch.setattr(
        "grdnet.pipeline.runner._maybe_load_checkpoint",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("grdnet.pipeline.runner.InferenceEngine", _InferenceEngineStub)
    assert runner.run_infer("cfg.yaml", checkpoint=None) == 0
    assert calls["calibrate"] == 1
    assert calls["infer"] == 1
