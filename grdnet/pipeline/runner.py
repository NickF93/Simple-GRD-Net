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
    profile = getattr(cfg, "profile", None)
    backend_cfg = getattr(cfg, "backend", None)
    system_cfg = getattr(cfg, "system", None)

    profile_name = getattr(profile, "name", "<unknown>")
    profile_mode = getattr(profile, "mode", "<unknown>")
    backend_name = getattr(backend_cfg, "name", "<unknown>")
    seed = getattr(system_cfg, "seed", "<unknown>")
    deterministic = getattr(system_cfg, "deterministic", "<unknown>")

    LOGGER.info("config_loaded path=%s", config_path)
    LOGGER.info(
        "profile_selected name=%s mode=%s backend=%s",
        profile_name,
        profile_mode,
        backend_name,
    )
    LOGGER.info(
        "reproducibility seed=%s deterministic=%s",
        seed,
        deterministic,
    )
    set_global_seed(seed=cfg.system.seed, deterministic=cfg.system.deterministic)

    LOGGER.info("initializing_backend")
    backend = create_backend(cfg)
    LOGGER.info("initializing_datamodule")
    datamodule = DataModule(cfg.data)
    LOGGER.info("initializing_reporters")
    reporters = [ConsoleReporter(), CsvReporter(cfg)]
    LOGGER.info("setup_completed")
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
    """Validate one YAML configuration file."""
    LOGGER.info("command=validate-config start path=%s", config_path)
    _ = load_experiment_config(config_path)
    LOGGER.info("command=validate-config success path=%s", config_path)
    return 0


def run_train(config_path: str) -> int:
    """Run end-to-end training using one profile configuration."""
    LOGGER.info("command=train start path=%s", config_path)
    cfg, backend, datamodule, reporters = _setup(config_path)

    LOGGER.info("building_train_loader")
    train_loader = datamodule.train_loader()
    LOGGER.info("building_val_loader")
    val_loader = datamodule.val_loader()

    engine = TrainingEngine(cfg=cfg, backend=backend, reporters=reporters)
    LOGGER.info("starting_training_engine")
    engine.train(train_loader, val_loader)
    LOGGER.info("command=train completed")
    return 0


def run_calibrate(config_path: str, checkpoint: str | None) -> int:
    """Calibrate anomaly threshold from the configured calibration split."""
    LOGGER.info(
        "command=calibrate start path=%s checkpoint=%s",
        config_path,
        checkpoint,
    )
    cfg, backend, datamodule, reporters = _setup(config_path)
    _maybe_load_checkpoint(backend, checkpoint)

    LOGGER.info("building_calibration_loader")
    calibration_loader = datamodule.calibration_loader()
    if calibration_loader is None:
        raise ValueError("data.calibration_dir must be set for calibration command")

    engine = InferenceEngine(cfg=cfg, backend=backend, reporters=reporters)
    LOGGER.info("running_calibration")
    threshold = engine.calibrate(calibration_loader)
    LOGGER.info("calibrated_threshold=%.8f", threshold)
    LOGGER.info("command=calibrate completed")
    return 0


def run_evaluate(config_path: str, checkpoint: str | None) -> int:
    """Evaluate the model and threshold on the configured test split."""
    LOGGER.info(
        "command=eval start path=%s checkpoint=%s",
        config_path,
        checkpoint,
    )
    cfg, backend, datamodule, reporters = _setup(config_path)
    _maybe_load_checkpoint(backend, checkpoint)

    LOGGER.info("building_test_loader")
    test_loader = datamodule.test_loader()
    if test_loader is None:
        raise ValueError("data.test_dir must be set for evaluation")

    inference_engine = InferenceEngine(cfg=cfg, backend=backend, reporters=reporters)

    threshold = cfg.inference.anomaly_threshold
    if threshold is None:
        LOGGER.info("threshold_missing calibrating_from_calibration_split")
        calibration_loader = datamodule.calibration_loader()
        if calibration_loader is None:
            raise ValueError(
                "No threshold available: set inference.anomaly_threshold "
                "or provide data.calibration_dir"
            )
        threshold = inference_engine.calibrate(calibration_loader)

    LOGGER.info("running_evaluation threshold=%s", threshold)
    metrics = inference_engine.evaluate(test_loader, threshold=threshold)
    LOGGER.info("evaluation_metrics=%s", metrics)
    LOGGER.info("command=eval completed")
    return 0


def run_infer(config_path: str, checkpoint: str | None) -> int:
    """Run inference and export prediction CSV rows."""
    LOGGER.info(
        "command=infer start path=%s checkpoint=%s",
        config_path,
        checkpoint,
    )
    cfg, backend, datamodule, reporters = _setup(config_path)
    _maybe_load_checkpoint(backend, checkpoint)

    LOGGER.info("building_test_loader")
    test_loader = datamodule.test_loader()
    if test_loader is None:
        raise ValueError("data.test_dir must be set for inference")

    inference_engine = InferenceEngine(cfg=cfg, backend=backend, reporters=reporters)

    threshold = cfg.inference.anomaly_threshold
    if threshold is None:
        LOGGER.info("threshold_missing calibrating_from_calibration_split")
        calibration_loader = datamodule.calibration_loader()
        if calibration_loader is None:
            raise ValueError(
                "No threshold available: set inference.anomaly_threshold "
                "or provide data.calibration_dir"
            )
        threshold = inference_engine.calibrate(calibration_loader)

    LOGGER.info("running_inference threshold=%s", threshold)
    prediction_count = inference_engine.infer(test_loader, threshold=threshold)
    LOGGER.info("predictions=%d", prediction_count)
    LOGGER.info("command=infer completed")
    return 0
