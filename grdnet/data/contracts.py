"""Data contracts and typed sample payloads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SampleItem:
    """Single dataset item contract."""

    image: torch.Tensor
    label: int
    path: str
    roi_mask: torch.Tensor
    gt_mask: torch.Tensor


class DatasetAdapter(Protocol):
    """Adapter interface for dataset providers."""

    def build_dataset(
        self,
        root: Path,
        *,
        nominal_only: bool,
        roi_root: Path | None,
        mask_root: Path | None,
    ) -> Dataset:
        """Build a dataset for a filesystem split."""
