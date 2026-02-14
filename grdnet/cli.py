"""Command line interface for GRD-Net."""

from __future__ import annotations

import argparse

from grdnet.core.runtime import enforce_runtime_versions
from grdnet.pipeline.runner import (
    run_calibrate,
    run_evaluate,
    run_infer,
    run_train,
    run_validate_config,
)


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


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and dispatch the selected command."""
    enforce_runtime_versions()
    parser = _build_parser()
    args = parser.parse_args(argv)

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
