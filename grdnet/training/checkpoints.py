"""Checkpoint serialization utilities."""

from __future__ import annotations

from pathlib import Path

import torch

from grdnet.backends.base import BackendStrategy


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
    payload = torch.load(path, map_location=backend.device)

    backend.models.generator.load_state_dict(payload["models"]["generator"])
    backend.models.discriminator.load_state_dict(payload["models"]["discriminator"])
    if backend.models.segmentator is not None and payload["models"]["segmentator"] is not None:
        backend.models.segmentator.load_state_dict(payload["models"]["segmentator"])

    backend.optimizers.generator.load_state_dict(payload["optimizers"]["generator"])
    backend.optimizers.discriminator.load_state_dict(payload["optimizers"]["discriminator"])
    if backend.optimizers.segmentator is not None and payload["optimizers"]["segmentator"] is not None:
        backend.optimizers.segmentator.load_state_dict(payload["optimizers"]["segmentator"])

    backend.schedulers.generator.load_state_dict(payload["schedulers"]["generator"])
    backend.schedulers.discriminator.load_state_dict(payload["schedulers"]["discriminator"])
    if backend.schedulers.segmentator is not None and payload["schedulers"]["segmentator"] is not None:
        backend.schedulers.segmentator.load_state_dict(payload["schedulers"]["segmentator"])

    return int(payload["epoch"])
