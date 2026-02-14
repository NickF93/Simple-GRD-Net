"""Command line interface for GRD-Net."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml

from grdnet.core.determinism import (
    CUBLAS_WORKSPACE_ALLOWED,
    CUBLAS_WORKSPACE_DEFAULT,
    CUBLAS_WORKSPACE_ENV,
    is_valid_cublas_workspace,
)
from grdnet.core.exceptions import ConfigurationError


def enforce_runtime_versions() -> None:
    """Lazily import runtime guard to keep Torch import out of module import path."""
    from grdnet.core.runtime import enforce_runtime_versions as _guard

    _guard()


def run_validate_config(config_path: str) -> int:
    """Lazily dispatch validate-config command."""
    from grdnet.pipeline.runner import run_validate_config as _run_validate_config

    return _run_validate_config(config_path)


def run_train(config_path: str) -> int:
    """Lazily dispatch train command."""
    from grdnet.pipeline.runner import run_train as _run_train

    return _run_train(config_path)


def run_calibrate(config_path: str, checkpoint: str | None) -> int:
    """Lazily dispatch calibrate command."""
    from grdnet.pipeline.runner import run_calibrate as _run_calibrate

    return _run_calibrate(config_path, checkpoint)


def run_evaluate(config_path: str, checkpoint: str | None) -> int:
    """Lazily dispatch eval command."""
    from grdnet.pipeline.runner import run_evaluate as _run_evaluate

    return _run_evaluate(config_path, checkpoint)


def run_infer(config_path: str, checkpoint: str | None) -> int:
    """Lazily dispatch infer command."""
    from grdnet.pipeline.runner import run_infer as _run_infer

    return _run_infer(config_path, checkpoint)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grdnet")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="Validate YAML config")
    validate.add_argument("-c", "--config", required=True, help="Path to YAML config")

    train = subparsers.add_parser("train", help="Train according to profile config")
    train.add_argument("-c", "--config", required=True, help="Path to YAML config")

    calibrate = subparsers.add_parser(
        "calibrate",
        help="Calibrate threshold on calibration split",
    )
    calibrate.add_argument("-c", "--config", required=True, help="Path to YAML config")
    calibrate.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path",
    )

    evaluate = subparsers.add_parser(
        "eval",
        help="Evaluate thresholded metrics on test split",
    )
    evaluate.add_argument("-c", "--config", required=True, help="Path to YAML config")
    evaluate.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path",
    )

    infer = subparsers.add_parser("infer", help="Run inference and export predictions")
    infer.add_argument("-c", "--config", required=True, help="Path to YAML config")
    infer.add_argument("--checkpoint", default=None, help="Optional checkpoint path")

    return parser


def _load_raw_payload(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ConfigurationError(
            f"Invalid config payload in {path}: top-level YAML node must be a mapping."
        )
    return payload


def _is_cuda_target(device: str) -> bool:
    candidate = device.strip().lower()
    if candidate == "auto":
        return True
    return candidate.startswith("cuda")


def _configure_determinism_environment(command: str, config_path: str) -> None:
    if command not in {"train", "calibrate", "eval", "infer"}:
        return

    payload = _load_raw_payload(config_path)
    if not payload:
        return

    system_cfg = payload.get("system")
    backend_cfg = payload.get("backend")
    if not isinstance(system_cfg, dict) or not isinstance(backend_cfg, dict):
        return

    deterministic = bool(system_cfg.get("deterministic", False))
    backend_name = str(backend_cfg.get("name", "pytorch"))
    backend_device = str(backend_cfg.get("device", "auto"))
    if not deterministic or backend_name != "pytorch" or not _is_cuda_target(
        backend_device
    ):
        return

    workspace = os.environ.get(CUBLAS_WORKSPACE_ENV)
    if workspace is None:
        os.environ[CUBLAS_WORKSPACE_ENV] = CUBLAS_WORKSPACE_DEFAULT
        return
    if not is_valid_cublas_workspace(workspace):
        allowed = ", ".join(sorted(CUBLAS_WORKSPACE_ALLOWED))
        raise ConfigurationError(
            "system.deterministic=true with CUDA requires "
            f"{CUBLAS_WORKSPACE_ENV} to be one of: {allowed}. "
            f"Current value: {workspace}"
        )


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and dispatch the selected command."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_determinism_environment(args.command, args.config)
    enforce_runtime_versions()

    if args.command == "validate-config":
        return run_validate_config(args.config)
    if args.command == "train":
        return run_train(args.config)
    if args.command == "calibrate":
        return run_calibrate(args.config, args.checkpoint)
    if args.command == "eval":
        return run_evaluate(args.config, args.checkpoint)
    if args.command == "infer":
        return run_infer(args.config, args.checkpoint)

    parser.error(f"Unsupported command: {args.command}")
    return 2
