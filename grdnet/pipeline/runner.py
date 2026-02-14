"""Application runners for CLI commands."""

from __future__ import annotations

import logging
from pathlib import Path

from grdnet.backends import create_backend
from grdnet.backends.base import BackendStrategy
from grdnet.config.loader import load_experiment_config
from grdnet.config.schema import ExperimentConfig
from grdnet.core.logging import configure_logging
from grdnet.core.reproducibility import set_global_seed, validate_determinism_runtime
from grdnet.data import DataModule
from grdnet.inference import InferenceEngine
from grdnet.reporting import ConsoleReporter, CsvReporter
from grdnet.reporting.base import Reporter
from grdnet.training import TrainingEngine
from grdnet.training.checkpoints import load_checkpoint

LOGGER = logging.getLogger(__name__)
_ANSI_CYAN = "\033[36m"
_ANSI_RESET = "\033[0m"


def _is_mvtec_category_root(root: Path) -> bool:
    return (
        root.is_dir()
        and (root / "train").is_dir()
        and (root / "test").is_dir()
        and (root / "ground_truth").is_dir()
    )


def _mvtec_benchmark_categories(root: Path | None) -> list[str]:
    if root is None or not root.exists() or not root.is_dir():
        return []
    categories: list[str] = []
    for child in sorted(root.iterdir()):
        if _is_mvtec_category_root(child):
            categories.append(child.name)
    return categories


def _scope_path_to_category(path: Path | None, category: str) -> Path | None:
    """Scope a split root to one category when `path` is an MVTec benchmark root."""
    if path is None:
        return None
    categories = _mvtec_benchmark_categories(path)
    if category in categories:
        return path / category
    return path


def _scope_required_path_to_category(path: Path, category: str) -> Path:
    """Scope a required path field and fail fast if it unexpectedly becomes null."""
    scoped = _scope_path_to_category(path, category)
    if scoped is None:
        raise ValueError("Required path unexpectedly resolved to None.")
    return scoped


def _is_mvtec_split_dir(path: Path) -> bool:
    return path.name in {"train", "test", "ground_truth"}


def _infer_category_root(cfg: ExperimentConfig) -> Path | None:
    candidates = (
        cfg.data.train_dir,
        cfg.data.test_dir,
        cfg.data.mask_dir,
    )
    for candidate in candidates:
        if candidate is None:
            continue
        if _is_mvtec_category_root(candidate):
            return candidate
        parent = candidate.parent
        if _is_mvtec_split_dir(candidate) and _is_mvtec_category_root(parent):
            return parent
    return None


def _display_path(path: Path | None) -> str:
    if path is None:
        return "<none>"
    return str(path)


def _log_benchmark_category_paths(
    *,
    command: str,
    category: str,
    cfg: ExperimentConfig,
) -> None:
    root = _infer_category_root(cfg)
    LOGGER.info(
        "%sbenchmark_category_paths command=%s category=%s root=%s train=%s "
        "test=%s mask=%s roi=%s%s",
        _ANSI_CYAN,
        command,
        category,
        _display_path(root),
        _display_path(cfg.data.train_dir),
        _display_path(cfg.data.test_dir),
        _display_path(cfg.data.mask_dir),
        _display_path(cfg.data.roi_dir),
        _ANSI_RESET,
    )


def _scoped_category_cfg(cfg: ExperimentConfig, category: str) -> ExperimentConfig:
    scoped = cfg.model_copy(deep=True)
    scoped.profile.name = f"{cfg.profile.name} [{category}]"
    scoped.data.train_dir = _scope_required_path_to_category(
        scoped.data.train_dir,
        category,
    )
    scoped.data.val_dir = _scope_path_to_category(scoped.data.val_dir, category)
    scoped.data.test_dir = _scope_path_to_category(scoped.data.test_dir, category)
    scoped.data.calibration_dir = _scope_path_to_category(
        scoped.data.calibration_dir,
        category,
    )
    scoped.data.roi_dir = _scope_path_to_category(scoped.data.roi_dir, category)
    scoped.data.mask_dir = _scope_path_to_category(scoped.data.mask_dir, category)
    scoped.training.checkpoint_dir = scoped.training.checkpoint_dir / category
    scoped.training.output_dir = scoped.training.output_dir / category
    return scoped


def _setup_from_cfg(
    cfg: ExperimentConfig,
    *,
    config_path: str,
    config_source: str = "scoped",
) -> tuple[ExperimentConfig, BackendStrategy, DataModule, list[Reporter]]:
    configure_logging(cfg.system.log_level)
    profile = getattr(cfg, "profile", None)
    backend_cfg = getattr(cfg, "backend", None)
    system_cfg = getattr(cfg, "system", None)

    profile_name = getattr(profile, "name", "<unknown>")
    profile_mode = getattr(profile, "mode", "<unknown>")
    backend_name = getattr(backend_cfg, "name", "<unknown>")
    seed = getattr(system_cfg, "seed", "<unknown>")
    deterministic = getattr(system_cfg, "deterministic", "<unknown>")

    if config_source == "file":
        LOGGER.info("config_loaded path=%s", config_path)
    else:
        LOGGER.info("config_reused source=%s path=%s", config_source, config_path)
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
    validate_determinism_runtime(
        deterministic=bool(getattr(system_cfg, "deterministic", False)),
        backend_name=str(getattr(backend_cfg, "name", "<unknown>")),
        backend_device=getattr(backend, "device", None),
    )
    LOGGER.info("initializing_datamodule")
    datamodule = DataModule(cfg.data)
    LOGGER.info("initializing_reporters")
    reporters = [ConsoleReporter(), CsvReporter(cfg)]
    LOGGER.info("setup_completed")
    return cfg, backend, datamodule, reporters


def _setup(
    config_path: str,
) -> tuple[ExperimentConfig, BackendStrategy, DataModule, list[Reporter]]:
    cfg = load_experiment_config(config_path)
    return _setup_from_cfg(cfg, config_path=config_path, config_source="file")


def _maybe_load_checkpoint(
    backend: BackendStrategy,
    checkpoint: str | None,
) -> None:
    if checkpoint is None:
        LOGGER.warning("No checkpoint provided. Using current model initialization.")
        return
    epoch = load_checkpoint(backend, Path(checkpoint))
    LOGGER.info("Loaded checkpoint from epoch %d", epoch)


def _resolve_category_checkpoint(
    checkpoint: str | None,
    *,
    category: str,
) -> str | None:
    if checkpoint is None:
        return None
    if "{category}" in checkpoint:
        return checkpoint.format(category=category)

    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_dir():
        category_dir = checkpoint_path / category
        candidates = sorted(category_dir.glob("epoch_*.pt"))
        if candidates:
            return str(candidates[-1])
        raise ValueError(
            "No category checkpoint found. Expected at least one file matching "
            f"'epoch_*.pt' in {category_dir}"
        )

    raise ValueError(
        "Benchmark mode requires --checkpoint to be either a directory "
        "containing category subfolders or a template string with "
        "'{category}'."
    )


def _cfg_categories(cfg: ExperimentConfig, *, split: str) -> list[str]:
    data = getattr(cfg, "data", None)
    if data is None:
        return []
    if split == "train":
        return _mvtec_benchmark_categories(getattr(data, "train_dir", None))
    if split == "test":
        return _mvtec_benchmark_categories(getattr(data, "test_dir", None))
    if split == "calibration":
        return _mvtec_benchmark_categories(getattr(data, "calibration_dir", None))
    return []


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
    categories = _cfg_categories(cfg, split="train")
    if categories:
        LOGGER.info(
            "benchmark_mode command=train categories=%s",
            ",".join(categories),
        )
        for category in categories:
            LOGGER.info("benchmark_category_start command=train category=%s", category)
            scoped_cfg = _scoped_category_cfg(cfg, category)
            _log_benchmark_category_paths(
                command="train",
                category=category,
                cfg=scoped_cfg,
            )
            scoped_cfg, backend, datamodule, reporters = _setup_from_cfg(
                scoped_cfg,
                config_path=config_path,
                config_source=f"benchmark_category:{category}",
            )
            LOGGER.info("building_train_loader")
            train_loader = datamodule.train_loader()
            LOGGER.info("building_val_loader")
            val_loader = datamodule.val_loader()
            engine = TrainingEngine(
                cfg=scoped_cfg,
                backend=backend,
                reporters=reporters,
            )
            LOGGER.info("starting_training_engine")
            engine.train(train_loader, val_loader)
            LOGGER.info("benchmark_category_done command=train category=%s", category)
        LOGGER.info("command=train completed")
        return 0

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
    categories = _cfg_categories(cfg, split="calibration")
    if categories:
        LOGGER.info(
            "benchmark_mode command=calibrate categories=%s",
            ",".join(categories),
        )
        for category in categories:
            LOGGER.info(
                "benchmark_category_start command=calibrate category=%s",
                category,
            )
            scoped_cfg = _scoped_category_cfg(cfg, category)
            _log_benchmark_category_paths(
                command="calibrate",
                category=category,
                cfg=scoped_cfg,
            )
            scoped_cfg, backend, datamodule, reporters = _setup_from_cfg(
                scoped_cfg,
                config_path=config_path,
                config_source=f"benchmark_category:{category}",
            )
            category_checkpoint = _resolve_category_checkpoint(
                checkpoint,
                category=category,
            )
            _maybe_load_checkpoint(backend, category_checkpoint)
            LOGGER.info("building_calibration_loader")
            calibration_loader = datamodule.calibration_loader()
            if calibration_loader is None:
                raise ValueError(
                    "data.calibration_dir must be set for calibration command"
                )
            engine = InferenceEngine(
                cfg=scoped_cfg,
                backend=backend,
                reporters=reporters,
            )
            LOGGER.info("running_calibration")
            threshold = engine.calibrate(calibration_loader)
            LOGGER.info(
                "calibrated_threshold category=%s value=%.8f",
                category,
                threshold,
            )
            LOGGER.info(
                "benchmark_category_done command=calibrate category=%s",
                category,
            )
        LOGGER.info("command=calibrate completed")
        return 0

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
    categories = _cfg_categories(cfg, split="test")
    if categories:
        LOGGER.info(
            "benchmark_mode command=eval categories=%s",
            ",".join(categories),
        )
        for category in categories:
            LOGGER.info("benchmark_category_start command=eval category=%s", category)
            scoped_cfg = _scoped_category_cfg(cfg, category)
            _log_benchmark_category_paths(
                command="eval",
                category=category,
                cfg=scoped_cfg,
            )
            scoped_cfg, backend, datamodule, reporters = _setup_from_cfg(
                scoped_cfg,
                config_path=config_path,
                config_source=f"benchmark_category:{category}",
            )
            category_checkpoint = _resolve_category_checkpoint(
                checkpoint,
                category=category,
            )
            _maybe_load_checkpoint(backend, category_checkpoint)

            LOGGER.info("building_test_loader")
            test_loader = datamodule.test_loader()
            if test_loader is None:
                raise ValueError("data.test_dir must be set for evaluation")

            inference_engine = InferenceEngine(
                cfg=scoped_cfg,
                backend=backend,
                reporters=reporters,
            )

            threshold = scoped_cfg.inference.anomaly_threshold
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
            LOGGER.info("evaluation_metrics category=%s metrics=%s", category, metrics)
            LOGGER.info("benchmark_category_done command=eval category=%s", category)
        LOGGER.info("command=eval completed")
        return 0

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
    categories = _cfg_categories(cfg, split="test")
    if categories:
        LOGGER.info(
            "benchmark_mode command=infer categories=%s",
            ",".join(categories),
        )
        for category in categories:
            LOGGER.info("benchmark_category_start command=infer category=%s", category)
            scoped_cfg = _scoped_category_cfg(cfg, category)
            _log_benchmark_category_paths(
                command="infer",
                category=category,
                cfg=scoped_cfg,
            )
            scoped_cfg, backend, datamodule, reporters = _setup_from_cfg(
                scoped_cfg,
                config_path=config_path,
                config_source=f"benchmark_category:{category}",
            )
            category_checkpoint = _resolve_category_checkpoint(
                checkpoint,
                category=category,
            )
            _maybe_load_checkpoint(backend, category_checkpoint)

            LOGGER.info("building_test_loader")
            test_loader = datamodule.test_loader()
            if test_loader is None:
                raise ValueError("data.test_dir must be set for inference")

            inference_engine = InferenceEngine(
                cfg=scoped_cfg,
                backend=backend,
                reporters=reporters,
            )

            threshold = scoped_cfg.inference.anomaly_threshold
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
            LOGGER.info("predictions category=%s count=%d", category, prediction_count)
            LOGGER.info("benchmark_category_done command=infer category=%s", category)
        LOGGER.info("command=infer completed")
        return 0

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
