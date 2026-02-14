"""Application runners for CLI commands."""

from __future__ import annotations

import logging
from pathlib import Path

from grdnet.backends import create_backend
from grdnet.backends.base import BackendStrategy
from grdnet.config.loader import load_experiment_config
from grdnet.config.schema import ExperimentConfig
from grdnet.core.logging import configure_logging
from grdnet.core.reproducibility import set_global_seed
from grdnet.data import DataModule
from grdnet.inference import InferenceEngine
from grdnet.reporting import ConsoleReporter, CsvReporter
from grdnet.reporting.base import Reporter
from grdnet.training import TrainingEngine
from grdnet.training.checkpoints import load_checkpoint

LOGGER = logging.getLogger(__name__)


def _setup(
    config_path: str,
) -> tuple[ExperimentConfig, BackendStrategy, DataModule, list[Reporter]]:
    cfg = load_experiment_config(config_path)
    configure_logging(cfg.system.log_level)
    set_global_seed(seed=cfg.system.seed, deterministic=cfg.system.deterministic)

    backend = create_backend(cfg)
    datamodule = DataModule(cfg.data)
    reporters = [ConsoleReporter(), CsvReporter(cfg)]
    return cfg, backend, datamodule, reporters


def _maybe_load_checkpoint(
    backend: BackendStrategy,
    checkpoint: str | None,
) -> None:
    if checkpoint is None:
        LOGGER.warning("No checkpoint provided. Using current model initialization.")
        return
    epoch = load_checkpoint(backend, Path(checkpoint))
    LOGGER.info("Loaded checkpoint from epoch %d", epoch)


def run_validate_config(config_path: str) -> int:
    _ = load_experiment_config(config_path)
    LOGGER.info("Config OK: %s", config_path)
    return 0


def run_train(config_path: str) -> int:
    cfg, backend, datamodule, reporters = _setup(config_path)

    train_loader = datamodule.train_loader()
    val_loader = datamodule.val_loader()

    engine = TrainingEngine(cfg=cfg, backend=backend, reporters=reporters)
    engine.train(train_loader, val_loader)
    return 0


def run_calibrate(config_path: str, checkpoint: str | None) -> int:
    cfg, backend, datamodule, reporters = _setup(config_path)
    _maybe_load_checkpoint(backend, checkpoint)

    calibration_loader = datamodule.calibration_loader()
    if calibration_loader is None:
        raise ValueError("data.calibration_dir must be set for calibration command")

    engine = InferenceEngine(cfg=cfg, backend=backend, reporters=reporters)
    threshold = engine.calibrate(calibration_loader)
    LOGGER.info("calibrated_threshold=%.8f", threshold)
    return 0


def run_evaluate(config_path: str, checkpoint: str | None) -> int:
    cfg, backend, datamodule, reporters = _setup(config_path)
    _maybe_load_checkpoint(backend, checkpoint)

    test_loader = datamodule.test_loader()
    if test_loader is None:
        raise ValueError("data.test_dir must be set for evaluation")

    inference_engine = InferenceEngine(cfg=cfg, backend=backend, reporters=reporters)

    threshold = cfg.inference.anomaly_threshold
    if threshold is None:
        calibration_loader = datamodule.calibration_loader()
        if calibration_loader is None:
            raise ValueError(
                "No threshold available: set inference.anomaly_threshold "
                "or provide data.calibration_dir"
            )
        threshold = inference_engine.calibrate(calibration_loader)

    metrics = inference_engine.evaluate(test_loader, threshold=threshold)
    LOGGER.info("evaluation_metrics=%s", metrics)
    return 0


def run_infer(config_path: str, checkpoint: str | None) -> int:
    cfg, backend, datamodule, reporters = _setup(config_path)
    _maybe_load_checkpoint(backend, checkpoint)

    test_loader = datamodule.test_loader()
    if test_loader is None:
        raise ValueError("data.test_dir must be set for inference")

    inference_engine = InferenceEngine(cfg=cfg, backend=backend, reporters=reporters)

    threshold = cfg.inference.anomaly_threshold
    if threshold is None:
        calibration_loader = datamodule.calibration_loader()
        if calibration_loader is None:
            raise ValueError(
                "No threshold available: set inference.anomaly_threshold "
                "or provide data.calibration_dir"
            )
        threshold = inference_engine.calibrate(calibration_loader)

    rows = inference_engine.infer(test_loader, threshold=threshold)
    LOGGER.info("predictions=%d", len(rows))
    return 0
