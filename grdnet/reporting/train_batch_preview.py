"""Training batch preview artifact writer."""

from __future__ import annotations

from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from grdnet.backends.base import TrainBatchPreview


class TrainBatchPreviewWriter:
    """Persist deterministic train-batch preview composites to disk."""

    def __init__(self, *, output_dir: Path, subdir: str, max_images: int) -> None:
        if max_images < 1:
            raise ValueError("max_images must be >= 1")
        if not subdir.strip():
            raise ValueError("subdir must be non-empty")

        relative_subdir = Path(subdir)
        if relative_subdir.is_absolute():
            raise ValueError("subdir must be a relative path")
        if ".." in relative_subdir.parts:
            raise ValueError("subdir must not contain parent path components")

        self._max_images = max_images
        self._output_dir = output_dir / relative_subdir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_tensor(name: str, value: torch.Tensor) -> None:
        if value.ndim != 4:
            shape = tuple(value.shape)
            raise ValueError(f"{name} must be 4D [B,C,H,W], got shape={shape}")
        if value.shape[0] < 1:
            raise ValueError(f"{name} batch dimension must be >= 1")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")

    @staticmethod
    def _to_rgb_batch(name: str, value: torch.Tensor, *, binary: bool) -> torch.Tensor:
        channel_count = value.shape[1]
        if channel_count in {1, 3}:
            out = value
        else:
            raise ValueError(f"{name} must have 1 or 3 channels, got {channel_count}")

        if binary:
            out = (out > 0.5).to(dtype=torch.float32)
        else:
            out = out.to(dtype=torch.float32).clamp(0.0, 1.0)

        if out.shape[1] == 1:
            out = out.repeat(1, 3, 1, 1)
        return out

    @staticmethod
    def _validate_aligned_shapes(preview: TrainBatchPreview) -> None:
        reference_shape = preview.x.shape
        expected_hw = reference_shape[2:]

        for name, value in (
            ("x_noisy", preview.x_noisy),
            ("noise_mask", preview.noise_mask),
            ("x_rebuilt", preview.x_rebuilt),
        ):
            if value.shape[0] != reference_shape[0]:
                actual = value.shape[0]
                expected = reference_shape[0]
                raise ValueError(
                    f"{name} batch size mismatch: {actual} != {expected}"
                )
            if value.shape[2:] != expected_hw:
                raise ValueError(
                    f"{name} spatial shape mismatch: {value.shape[2:]} != {expected_hw}"
                )

    def write(self, *, epoch: int, step: int, preview: TrainBatchPreview) -> Path:
        """Write one 4-row preview composite and return its path."""
        if epoch < 1:
            raise ValueError("epoch must be >= 1")
        if step < 1:
            raise ValueError("step must be >= 1")

        self._validate_tensor("x", preview.x)
        self._validate_tensor("x_noisy", preview.x_noisy)
        self._validate_tensor("noise_mask", preview.noise_mask)
        self._validate_tensor("x_rebuilt", preview.x_rebuilt)
        self._validate_aligned_shapes(preview)

        x = preview.x.detach().cpu()
        x_noisy = preview.x_noisy.detach().cpu()
        noise_mask = preview.noise_mask.detach().cpu()
        x_rebuilt = preview.x_rebuilt.detach().cpu()

        batch_count = min(self._max_images, x.shape[0])
        rows = (
            self._to_rgb_batch("x", x[:batch_count], binary=False),
            self._to_rgb_batch("x_noisy", x_noisy[:batch_count], binary=False),
            self._to_rgb_batch("noise_mask", noise_mask[:batch_count], binary=True),
            self._to_rgb_batch("x_rebuilt", x_rebuilt[:batch_count], binary=False),
        )
        row_grids = [
            make_grid(row, nrow=batch_count, padding=2, normalize=False) for row in rows
        ]
        composite = torch.cat(row_grids, dim=1)

        path = self._output_dir / f"epoch_{epoch:04d}_step_{step:04d}.png"
        save_image(composite, path)
        return path
