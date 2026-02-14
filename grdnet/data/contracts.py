"""Data contracts and typed sample payloads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias

import torch
from torch.utils.data import Dataset

SampleValue: TypeAlias = torch.Tensor | int | str
SampleBatchItem: TypeAlias = dict[str, SampleValue]


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
        mask_enabled: bool = True,
    ) -> Dataset[SampleBatchItem]:
        """Build a dataset for a filesystem split."""
