"""PyTorch dataloader orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from grdnet.config.schema import DataConfig
from grdnet.core.exceptions import DatasetContractError
from grdnet.data.adapters.mvtec import MvtecLikeAdapter


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

    def _build_dataset(
        self,
        root: Path,
        *,
        nominal_only: bool,
        use_roi: bool,
        use_mask: bool,
        split_kind: Literal["train", "val", "test", "calibration"],
    ) -> Dataset:
        roi_root = self._cfg.roi_dir if use_roi else None
        mask_root = self._cfg.mask_dir if use_mask else None

        if self._is_mvtec_category_root(root):
            split_root = root / self._split_subdir(split_kind)
            inferred_mask_root = root / "ground_truth" if use_mask else None
            effective_mask_root = (
                mask_root if mask_root is not None else inferred_mask_root
            )
            if use_mask and effective_mask_root is None:
                raise DatasetContractError(
                    "Unable to resolve mask root for MVTec category evaluation."
                )
            return self._adapter.build_dataset(
                root=split_root,
                nominal_only=nominal_only,
                roi_root=roi_root,
                mask_root=effective_mask_root,
            )

        categories = self._mvtec_benchmark_categories(root)
        if categories:
            datasets: list[Dataset] = []
            split_subdir = self._split_subdir(split_kind)
            for category_root in categories:
                split_root = category_root / split_subdir
                inferred_mask_root = (
                    category_root / "ground_truth" if use_mask else None
                )
                effective_mask_root = (
                    mask_root if mask_root is not None else inferred_mask_root
                )
                if use_mask and effective_mask_root is None:
                    raise DatasetContractError(
                        "Unable to resolve mask root for MVTec benchmark evaluation."
                    )
                datasets.append(
                    self._adapter.build_dataset(
                        root=split_root,
                        nominal_only=nominal_only,
                        roi_root=roi_root,
                        mask_root=effective_mask_root,
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
        )

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self._cfg.batch_size,
            shuffle=shuffle,
            num_workers=self._cfg.num_workers,
            # Default to warning-free behavior across CPU/GPU CI and local runs.
            pin_memory=False,
            drop_last=False,
        )

    def train_loader(self) -> DataLoader:
        dataset = self._build_dataset(
            root=self._cfg.train_dir,
            nominal_only=self._cfg.nominal_train_only,
            use_roi=True,
            use_mask=False,
            split_kind="train",
        )
        return self._loader(dataset, shuffle=True)

    def val_loader(self) -> DataLoader | None:
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
        train_size = int(0.95 * len(dataset))
        val_size = len(dataset) - train_size
        _, val_subset = random_split(dataset, [train_size, val_size])
        return self._loader(val_subset, shuffle=False)

    def test_loader(self) -> DataLoader | None:
        if self._cfg.test_dir is None:
            return None
        dataset = self._build_dataset(
            root=self._cfg.test_dir,
            nominal_only=False,
            use_roi=False,
            use_mask=True,
            split_kind="test",
        )
        return self._loader(dataset, shuffle=False)

    def calibration_loader(self) -> DataLoader | None:
        if self._cfg.calibration_dir is None:
            return None
        dataset = self._build_dataset(
            root=self._cfg.calibration_dir,
            nominal_only=False,
            use_roi=False,
            use_mask=True,
            split_kind="calibration",
        )
        return self._loader(dataset, shuffle=False)
