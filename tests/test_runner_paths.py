from pathlib import Path
from types import SimpleNamespace

import pytest

from grdnet.config.loader import load_experiment_config
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
            return 1

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


def test_run_train_benchmark_root_trains_one_model_per_category(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "mvtec"
    for category in ("bottle", "zipper"):
        (root / category / "train").mkdir(parents=True, exist_ok=True)
        (root / category / "test").mkdir(parents=True, exist_ok=True)
        (root / category / "ground_truth").mkdir(parents=True, exist_ok=True)

    cfg = load_experiment_config(Path("configs/profiles/deepindustrial_sn_2026.yaml"))
    cfg.data.train_dir = root
    cfg.data.test_dir = root
    cfg.data.calibration_dir = root
    cfg.training.checkpoint_dir = tmp_path / "checkpoints"
    cfg.training.output_dir = tmp_path / "reports"

    created_train_dirs: list[str] = []
    train_calls: list[str] = []

    class _DataModuleStub:
        def __init__(self, data_cfg) -> None:
            _ = data_cfg

        def train_loader(self):
            return [{"x": 1}]

        def val_loader(self):
            return [{"x": 1}]

    class _TrainingEngineStub:
        def __init__(self, *, cfg, backend, reporters) -> None:
            _ = backend, reporters
            train_calls.append(str(cfg.data.train_dir))

        def train(self, train_loader, val_loader) -> None:
            _ = train_loader, val_loader

    def _create_backend_spy(in_cfg):
        created_train_dirs.append(str(in_cfg.data.train_dir))
        return object()

    monkeypatch.setattr(
        "grdnet.pipeline.runner._setup",
        lambda config_path: (cfg, object(), object(), []),
    )
    monkeypatch.setattr("grdnet.pipeline.runner.configure_logging", lambda level: None)
    monkeypatch.setattr(
        "grdnet.pipeline.runner.set_global_seed",
        lambda seed, deterministic: None,
    )
    monkeypatch.setattr("grdnet.pipeline.runner.create_backend", _create_backend_spy)
    monkeypatch.setattr("grdnet.pipeline.runner.DataModule", _DataModuleStub)
    monkeypatch.setattr("grdnet.pipeline.runner.TrainingEngine", _TrainingEngineStub)
    monkeypatch.setattr("grdnet.pipeline.runner.ConsoleReporter", lambda: object())
    monkeypatch.setattr("grdnet.pipeline.runner.CsvReporter", lambda in_cfg: object())

    assert runner.run_train("cfg.yaml") == 0
    assert created_train_dirs == [
        str(root / "bottle"),
        str(root / "zipper"),
    ]
    assert train_calls == [
        str(root / "bottle"),
        str(root / "zipper"),
    ]


def test_run_calibrate_benchmark_root_resolves_category_checkpoints(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "mvtec"
    for category in ("bottle", "zipper"):
        (root / category / "train").mkdir(parents=True, exist_ok=True)
        (root / category / "test").mkdir(parents=True, exist_ok=True)
        (root / category / "ground_truth").mkdir(parents=True, exist_ok=True)

    checkpoint_root = tmp_path / "checkpoints"
    (checkpoint_root / "bottle").mkdir(parents=True, exist_ok=True)
    (checkpoint_root / "zipper").mkdir(parents=True, exist_ok=True)
    (checkpoint_root / "bottle" / "epoch_0007.pt").write_text("", encoding="utf-8")
    (checkpoint_root / "zipper" / "epoch_0011.pt").write_text("", encoding="utf-8")

    cfg = load_experiment_config(Path("configs/profiles/deepindustrial_sn_2026.yaml"))
    cfg.data.train_dir = root
    cfg.data.test_dir = root
    cfg.data.calibration_dir = root
    cfg.training.checkpoint_dir = tmp_path / "checkpoints_out"
    cfg.training.output_dir = tmp_path / "reports_out"

    loaded_checkpoints: list[str] = []

    class _DataModuleStub:
        def __init__(self, data_cfg) -> None:
            _ = data_cfg

        def calibration_loader(self):
            return [{"x": 1}]

    class _InferenceEngineStub:
        def __init__(self, *, cfg, backend, reporters) -> None:
            _ = cfg, backend, reporters

        def calibrate(self, loader) -> float:
            _ = loader
            return 0.2

    monkeypatch.setattr(
        "grdnet.pipeline.runner._setup",
        lambda config_path: (cfg, object(), object(), []),
    )
    monkeypatch.setattr("grdnet.pipeline.runner.configure_logging", lambda level: None)
    monkeypatch.setattr(
        "grdnet.pipeline.runner.set_global_seed",
        lambda seed, deterministic: None,
    )
    monkeypatch.setattr(
        "grdnet.pipeline.runner.create_backend",
        lambda in_cfg: object(),
    )
    monkeypatch.setattr("grdnet.pipeline.runner.DataModule", _DataModuleStub)
    monkeypatch.setattr("grdnet.pipeline.runner.InferenceEngine", _InferenceEngineStub)
    monkeypatch.setattr("grdnet.pipeline.runner.ConsoleReporter", lambda: object())
    monkeypatch.setattr("grdnet.pipeline.runner.CsvReporter", lambda in_cfg: object())
    monkeypatch.setattr(
        "grdnet.pipeline.runner._maybe_load_checkpoint",
        lambda backend, checkpoint: loaded_checkpoints.append(str(checkpoint)),
    )

    assert runner.run_calibrate("cfg.yaml", checkpoint=str(checkpoint_root)) == 0
    assert loaded_checkpoints == [
        str(checkpoint_root / "bottle" / "epoch_0007.pt"),
        str(checkpoint_root / "zipper" / "epoch_0011.pt"),
    ]


def test_run_evaluate_benchmark_root_runs_per_category(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "mvtec"
    for category in ("bottle", "zipper"):
        (root / category / "train").mkdir(parents=True, exist_ok=True)
        (root / category / "test").mkdir(parents=True, exist_ok=True)
        (root / category / "ground_truth").mkdir(parents=True, exist_ok=True)

    checkpoint_root = tmp_path / "checkpoints"
    (checkpoint_root / "bottle").mkdir(parents=True, exist_ok=True)
    (checkpoint_root / "zipper").mkdir(parents=True, exist_ok=True)
    (checkpoint_root / "bottle" / "epoch_0004.pt").write_text("", encoding="utf-8")
    (checkpoint_root / "zipper" / "epoch_0006.pt").write_text("", encoding="utf-8")

    cfg = load_experiment_config(Path("configs/profiles/deepindustrial_sn_2026.yaml"))
    cfg.data.train_dir = root
    cfg.data.test_dir = root
    cfg.data.calibration_dir = root
    cfg.training.checkpoint_dir = tmp_path / "checkpoints_out"
    cfg.training.output_dir = tmp_path / "reports_out"
    cfg.inference.anomaly_threshold = 0.33

    loaded_checkpoints: list[str] = []
    evaluated_categories: list[str] = []

    class _DataModuleStub:
        def __init__(self, data_cfg) -> None:
            self._data_cfg = data_cfg

        def test_loader(self):
            return [{"x": 1}]

        def calibration_loader(self):
            return [{"x": 1}]

    class _InferenceEngineStub:
        def __init__(self, *, cfg, backend, reporters) -> None:
            _ = backend, reporters
            self._cfg = cfg

        def calibrate(self, loader) -> float:
            _ = loader
            return 0.2

        def evaluate(self, loader, threshold):
            _ = loader, threshold
            evaluated_categories.append(self._cfg.profile.name)
            return {"accuracy": 1.0}

    monkeypatch.setattr(
        "grdnet.pipeline.runner._setup",
        lambda config_path: (cfg, object(), object(), []),
    )
    monkeypatch.setattr("grdnet.pipeline.runner.configure_logging", lambda level: None)
    monkeypatch.setattr(
        "grdnet.pipeline.runner.set_global_seed",
        lambda seed, deterministic: None,
    )
    monkeypatch.setattr(
        "grdnet.pipeline.runner.create_backend",
        lambda in_cfg: object(),
    )
    monkeypatch.setattr("grdnet.pipeline.runner.DataModule", _DataModuleStub)
    monkeypatch.setattr("grdnet.pipeline.runner.InferenceEngine", _InferenceEngineStub)
    monkeypatch.setattr("grdnet.pipeline.runner.ConsoleReporter", lambda: object())
    monkeypatch.setattr("grdnet.pipeline.runner.CsvReporter", lambda in_cfg: object())
    monkeypatch.setattr(
        "grdnet.pipeline.runner._maybe_load_checkpoint",
        lambda backend, checkpoint: loaded_checkpoints.append(str(checkpoint)),
    )

    assert runner.run_evaluate("cfg.yaml", checkpoint=str(checkpoint_root)) == 0
    assert loaded_checkpoints == [
        str(checkpoint_root / "bottle" / "epoch_0004.pt"),
        str(checkpoint_root / "zipper" / "epoch_0006.pt"),
    ]
    assert evaluated_categories == [
        "DeepIndustrial-SN 2026 Official [bottle]",
        "DeepIndustrial-SN 2026 Official [zipper]",
    ]


def test_run_infer_benchmark_root_runs_per_category(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "mvtec"
    for category in ("bottle", "zipper"):
        (root / category / "train").mkdir(parents=True, exist_ok=True)
        (root / category / "test").mkdir(parents=True, exist_ok=True)
        (root / category / "ground_truth").mkdir(parents=True, exist_ok=True)

    checkpoint_root = tmp_path / "checkpoints"
    (checkpoint_root / "bottle").mkdir(parents=True, exist_ok=True)
    (checkpoint_root / "zipper").mkdir(parents=True, exist_ok=True)
    (checkpoint_root / "bottle" / "epoch_0008.pt").write_text("", encoding="utf-8")
    (checkpoint_root / "zipper" / "epoch_0009.pt").write_text("", encoding="utf-8")

    cfg = load_experiment_config(Path("configs/profiles/deepindustrial_sn_2026.yaml"))
    cfg.data.train_dir = root
    cfg.data.test_dir = root
    cfg.data.calibration_dir = root
    cfg.training.checkpoint_dir = tmp_path / "checkpoints_out"
    cfg.training.output_dir = tmp_path / "reports_out"
    cfg.inference.anomaly_threshold = 0.33

    loaded_checkpoints: list[str] = []
    inferred_categories: list[str] = []

    class _DataModuleStub:
        def __init__(self, data_cfg) -> None:
            self._data_cfg = data_cfg

        def test_loader(self):
            return [{"x": 1}]

        def calibration_loader(self):
            return [{"x": 1}]

    class _InferenceEngineStub:
        def __init__(self, *, cfg, backend, reporters) -> None:
            _ = backend, reporters
            self._cfg = cfg

        def calibrate(self, loader) -> float:
            _ = loader
            return 0.2

        def infer(self, loader, threshold):
            _ = loader, threshold
            inferred_categories.append(self._cfg.profile.name)
            return 1

    monkeypatch.setattr(
        "grdnet.pipeline.runner._setup",
        lambda config_path: (cfg, object(), object(), []),
    )
    monkeypatch.setattr("grdnet.pipeline.runner.configure_logging", lambda level: None)
    monkeypatch.setattr(
        "grdnet.pipeline.runner.set_global_seed",
        lambda seed, deterministic: None,
    )
    monkeypatch.setattr(
        "grdnet.pipeline.runner.create_backend",
        lambda in_cfg: object(),
    )
    monkeypatch.setattr("grdnet.pipeline.runner.DataModule", _DataModuleStub)
    monkeypatch.setattr("grdnet.pipeline.runner.InferenceEngine", _InferenceEngineStub)
    monkeypatch.setattr("grdnet.pipeline.runner.ConsoleReporter", lambda: object())
    monkeypatch.setattr("grdnet.pipeline.runner.CsvReporter", lambda in_cfg: object())
    monkeypatch.setattr(
        "grdnet.pipeline.runner._maybe_load_checkpoint",
        lambda backend, checkpoint: loaded_checkpoints.append(str(checkpoint)),
    )

    assert runner.run_infer("cfg.yaml", checkpoint=str(checkpoint_root)) == 0
    assert loaded_checkpoints == [
        str(checkpoint_root / "bottle" / "epoch_0008.pt"),
        str(checkpoint_root / "zipper" / "epoch_0009.pt"),
    ]
    assert inferred_categories == [
        "DeepIndustrial-SN 2026 Official [bottle]",
        "DeepIndustrial-SN 2026 Official [zipper]",
    ]


def test_log_benchmark_category_paths_includes_expected_directories(
    monkeypatch,
    tmp_path: Path,
) -> None:
    category_root = tmp_path / "mvtec" / "bottle"
    (category_root / "train").mkdir(parents=True, exist_ok=True)
    (category_root / "test").mkdir(parents=True, exist_ok=True)
    (category_root / "ground_truth").mkdir(parents=True, exist_ok=True)
    (category_root / "roi").mkdir(parents=True, exist_ok=True)

    cfg = load_experiment_config(Path("configs/profiles/deepindustrial_sn_2026.yaml"))
    cfg.data.train_dir = category_root
    cfg.data.test_dir = category_root
    cfg.data.mask_dir = category_root / "ground_truth"
    cfg.data.roi_dir = category_root / "roi"

    emitted: list[str] = []

    def _capture(message: str, *args) -> None:
        emitted.append(message % args)

    monkeypatch.setattr("grdnet.pipeline.runner.LOGGER.info", _capture)
    runner._log_benchmark_category_paths(command="eval", category="bottle", cfg=cfg)

    assert len(emitted) == 1
    line = emitted[0]
    assert "\x1b[36m" in line
    assert "benchmark_category_paths command=eval category=bottle" in line
    assert f"root={category_root}" in line
    assert f"train={category_root}" in line
    assert f"test={category_root}" in line
    assert f"mask={category_root / 'ground_truth'}" in line
    assert f"roi={category_root / 'roi'}" in line


def test_log_benchmark_category_paths_root_inferred_from_split_dir(
    monkeypatch,
    tmp_path: Path,
) -> None:
    category_root = tmp_path / "mvtec" / "zipper"
    (category_root / "train").mkdir(parents=True, exist_ok=True)
    (category_root / "test").mkdir(parents=True, exist_ok=True)
    (category_root / "ground_truth").mkdir(parents=True, exist_ok=True)

    cfg = load_experiment_config(Path("configs/profiles/deepindustrial_sn_2026.yaml"))
    cfg.data.train_dir = category_root / "train"
    cfg.data.test_dir = category_root / "test"
    cfg.data.mask_dir = category_root / "ground_truth"
    cfg.data.roi_dir = None

    emitted: list[str] = []

    def _capture(message: str, *args) -> None:
        emitted.append(message % args)

    monkeypatch.setattr("grdnet.pipeline.runner.LOGGER.info", _capture)
    runner._log_benchmark_category_paths(command="infer", category="zipper", cfg=cfg)

    assert len(emitted) == 1
    line = emitted[0]
    assert f"root={category_root}" in line
    assert "roi=<none>" in line
