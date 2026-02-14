"""Checkpoint serialization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from grdnet.backends.base import BackendStrategy
from grdnet.core.exceptions import CheckpointError

_BUNDLE_KEYS = ("generator", "discriminator", "segmentator")


def _validate_checkpoint_payload(payload: Any, path: Path) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise CheckpointError(f"Invalid checkpoint at {path}: payload must be a dict")

    required_top_level = ("epoch", "models", "optimizers", "schedulers")
    for key in required_top_level:
        if key not in payload:
            raise CheckpointError(
                f"Invalid checkpoint at {path}: missing top-level key '{key}'"
            )

    epoch = payload["epoch"]
    if not isinstance(epoch, int):
        raise CheckpointError(f"Invalid checkpoint at {path}: 'epoch' must be int")
    if epoch < 0:
        raise CheckpointError(f"Invalid checkpoint at {path}: 'epoch' must be >= 0")

    for section in ("models", "optimizers", "schedulers"):
        section_payload = payload[section]
        if not isinstance(section_payload, dict):
            raise CheckpointError(
                f"Invalid checkpoint at {path}: section '{section}' must be dict"
            )
        for key in _BUNDLE_KEYS:
            if key not in section_payload:
                raise CheckpointError(
                    f"Invalid checkpoint at {path}: missing '{section}.{key}'"
                )

    return payload


def save_checkpoint(backend: BackendStrategy, path: Path, *, epoch: int) -> None:
    """Persist model/optimizer/scheduler state to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "models": {
            "generator": backend.models.generator.state_dict(),
            "discriminator": backend.models.discriminator.state_dict(),
            "segmentator": None
            if backend.models.segmentator is None
            else backend.models.segmentator.state_dict(),
        },
        "optimizers": {
            "generator": backend.optimizers.generator.state_dict(),
            "discriminator": backend.optimizers.discriminator.state_dict(),
            "segmentator": None
            if backend.optimizers.segmentator is None
            else backend.optimizers.segmentator.state_dict(),
        },
        "schedulers": {
            "generator": backend.schedulers.generator.state_dict(),
            "discriminator": backend.schedulers.discriminator.state_dict(),
            "segmentator": None
            if backend.schedulers.segmentator is None
            else backend.schedulers.segmentator.state_dict(),
        },
    }
    torch.save(state, path)


def load_checkpoint(backend: BackendStrategy, path: Path) -> int:
    """Load checkpoint and return last completed epoch."""
    if not path.exists():
        raise CheckpointError(f"Checkpoint not found: {path}")
    if not path.is_file():
        raise CheckpointError(f"Checkpoint path is not a file: {path}")

    try:
        raw_payload = torch.load(
            path,
            map_location=backend.device,
            weights_only=True,
        )
    except Exception as exc:
        raise CheckpointError(f"Unable to read checkpoint {path}: {exc}") from exc

    payload = _validate_checkpoint_payload(raw_payload, path)

    try:
        backend.models.generator.load_state_dict(payload["models"]["generator"])
        backend.models.discriminator.load_state_dict(payload["models"]["discriminator"])

        segmentator_model_state = payload["models"]["segmentator"]
        if backend.models.segmentator is not None:
            if segmentator_model_state is None:
                raise CheckpointError(
                    "Checkpoint missing model state for enabled segmentator."
                )
            backend.models.segmentator.load_state_dict(segmentator_model_state)

        backend.optimizers.generator.load_state_dict(payload["optimizers"]["generator"])
        backend.optimizers.discriminator.load_state_dict(
            payload["optimizers"]["discriminator"]
        )

        segmentator_opt_state = payload["optimizers"]["segmentator"]
        if backend.optimizers.segmentator is not None:
            if segmentator_opt_state is None:
                raise CheckpointError(
                    "Checkpoint missing optimizer state for enabled segmentator."
                )
            backend.optimizers.segmentator.load_state_dict(segmentator_opt_state)

        backend.schedulers.generator.load_state_dict(payload["schedulers"]["generator"])
        backend.schedulers.discriminator.load_state_dict(
            payload["schedulers"]["discriminator"]
        )

        segmentator_sched_state = payload["schedulers"]["segmentator"]
        if backend.schedulers.segmentator is not None:
            if segmentator_sched_state is None:
                raise CheckpointError(
                    "Checkpoint missing scheduler state for enabled segmentator."
                )
            backend.schedulers.segmentator.load_state_dict(segmentator_sched_state)
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError(
            f"Unable to restore checkpoint state from {path}: {exc}"
        ) from exc

    return int(payload["epoch"])
