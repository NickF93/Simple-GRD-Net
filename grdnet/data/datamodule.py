"""PyTorch dataloader orchestration."""

from __future__ import annotations

import logging
from collections.abc import Sized
from pathlib import Path
from typing import Literal

from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from grdnet.config.schema import DataConfig
from grdnet.core.exceptions import DatasetContractError
from grdnet.data.adapters.mvtec import MvtecLikeAdapter
from grdnet.data.contracts import SampleBatchItem

LOGGER = logging.getLogger(__name__)


class DataModule:
    """Build split-specific dataloaders from MVTec-like adapters."""

    def __init__(self, cfg: DataConfig) -> None:
        self._cfg = cfg
        self._adapter = MvtecLikeAdapter(cfg)

    @staticmethod
    def _is_mvtec_category_root(root: Path) -> bool:
        return (
            root.is_dir()
            and (root / "train").is_dir()
            and (root / "test").is_dir()
            and (root / "ground_truth").is_dir()
        )

    def _mvtec_benchmark_categories(self, root: Path) -> list[Path]:
        if not root.exists() or not root.is_dir():
            return []
        return [
            child
            for child in sorted(root.iterdir())
            if child.is_dir() and self._is_mvtec_category_root(child)
        ]

    @staticmethod
    def _split_subdir(
        split_kind: Literal["train", "val", "test", "calibration"],
    ) -> str:
        mapping = {
            "train": "train",
            "val": "train",
            "test": "test",
            "calibration": "test",
        }
        return mapping[split_kind]

    @staticmethod
    def _resolve_effective_mask_root(
        *,
        use_mask: bool,
        configured_mask_root: Path | None,
        inferred_mask_root: Path | None,
        missing_message: str,
    ) -> Path | None:
        """Resolve explicit/inferred GT mask root or fail fast when required."""
        if not use_mask:
            return None
        effective = (
            configured_mask_root
            if configured_mask_root is not None
            else inferred_mask_root
        )
        if effective is None:
            raise DatasetContractError(missing_message)
        return effective

    def _build_dataset(
        self,
        root: Path,
        *,
        nominal_only: bool,
        use_roi: bool,
        use_mask: bool,
        split_kind: Literal["train", "val", "test", "calibration"],
    ) -> Dataset[SampleBatchItem]:
        roi_root = self._cfg.roi_dir if use_roi else None
        mask_root = self._cfg.mask_dir if use_mask else None

        if use_roi and roi_root is not None and not roi_root.exists():
            LOGGER.warning(
                "roi_dir_not_found path=%s fallback=full_image_roi",
                roi_root,
            )

        if self._is_mvtec_category_root(root):
            split_root = root / self._split_subdir(split_kind)
            inferred_mask_root = root / "ground_truth" if use_mask else None
            effective_mask_root = self._resolve_effective_mask_root(
                use_mask=use_mask,
                configured_mask_root=mask_root,
                inferred_mask_root=inferred_mask_root,
                missing_message=(
                    "Unable to resolve mask root for MVTec category evaluation."
                ),
            )
            return self._adapter.build_dataset(
                root=split_root,
                nominal_only=nominal_only,
                roi_root=roi_root,
                mask_root=effective_mask_root,
                mask_enabled=use_mask,
            )

        categories = self._mvtec_benchmark_categories(root)
        if categories:
            datasets: list[Dataset[SampleBatchItem]] = []
            split_subdir = self._split_subdir(split_kind)
            for category_root in categories:
                split_root = category_root / split_subdir
                inferred_mask_root = (
                    category_root / "ground_truth" if use_mask else None
                )
                effective_mask_root = self._resolve_effective_mask_root(
                    use_mask=use_mask,
                    configured_mask_root=mask_root,
                    inferred_mask_root=inferred_mask_root,
                    missing_message=(
                        "Unable to resolve mask root for MVTec benchmark evaluation."
                    ),
                )
                datasets.append(
                    self._adapter.build_dataset(
                        root=split_root,
                        nominal_only=nominal_only,
                        roi_root=roi_root,
                        mask_root=effective_mask_root,
                        mask_enabled=use_mask,
                    )
                )
            if not datasets:
                raise DatasetContractError(
                    "No datasets could be built from MVTec benchmark root."
                )
            if len(datasets) == 1:
                return datasets[0]
            return ConcatDataset(datasets)

        return self._adapter.build_dataset(
            root=root,
            nominal_only=nominal_only,
            roi_root=roi_root,
            mask_root=mask_root,
            mask_enabled=use_mask,
        )

    def _loader(
        self,
        dataset: Dataset[SampleBatchItem],
        shuffle: bool,
    ) -> DataLoader[SampleBatchItem]:
        return DataLoader(
            dataset,
            batch_size=self._cfg.batch_size,
            shuffle=shuffle,
            num_workers=self._cfg.num_workers,
            # Default to warning-free behavior across CPU/GPU CI and local runs.
            pin_memory=False,
            drop_last=False,
        )

    def train_loader(self) -> DataLoader[SampleBatchItem]:
        """Build the training dataloader."""
        dataset = self._build_dataset(
            root=self._cfg.train_dir,
            nominal_only=self._cfg.nominal_train_only,
            use_roi=True,
            use_mask=False,
            split_kind="train",
        )
        return self._loader(dataset, shuffle=True)

    def val_loader(self) -> DataLoader[SampleBatchItem] | None:
        """Build validation loader or derive a split from training data."""
        if self._cfg.val_dir is not None:
            dataset = self._build_dataset(
                root=self._cfg.val_dir,
                nominal_only=self._cfg.nominal_train_only,
                use_roi=True,
                use_mask=False,
                split_kind="val",
            )
            return self._loader(dataset, shuffle=False)

        dataset = self._build_dataset(
            root=self._cfg.train_dir,
            nominal_only=self._cfg.nominal_train_only,
            use_roi=True,
            use_mask=False,
            split_kind="train",
        )
        if not isinstance(dataset, Sized):
            raise DatasetContractError("Training dataset must implement __len__().")
        dataset_len = len(dataset)
        train_size = int(0.95 * dataset_len)
        val_size = dataset_len - train_size
        _, val_subset = random_split(dataset, [train_size, val_size])
        return self._loader(val_subset, shuffle=False)

    def test_loader(self) -> DataLoader[SampleBatchItem] | None:
        """Build test/evaluation dataloader when configured."""
        if self._cfg.test_dir is None:
            return None
        dataset = self._build_dataset(
            root=self._cfg.test_dir,
            nominal_only=False,
            use_roi=True,
            use_mask=True,
            split_kind="test",
        )
        return self._loader(dataset, shuffle=False)

    def calibration_loader(self) -> DataLoader[SampleBatchItem] | None:
        """Build calibration dataloader when configured."""
        if self._cfg.calibration_dir is None:
            return None
        dataset = self._build_dataset(
            root=self._cfg.calibration_dir,
            nominal_only=False,
            use_roi=True,
            use_mask=True,
            split_kind="calibration",
        )
        return self._loader(dataset, shuffle=False)
