"""PyTorch dataloader orchestration."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader, Dataset, random_split

from grdnet.config.schema import DataConfig
from grdnet.data.adapters.mvtec import MvtecLikeAdapter


class DataModule:
    """Build split-specific dataloaders from MVTec-like adapters."""

    def __init__(self, cfg: DataConfig) -> None:
        self._cfg = cfg
        self._adapter = MvtecLikeAdapter(cfg)

    def _build_dataset(
        self,
        root: Path,
        *,
        nominal_only: bool,
        use_roi: bool,
        use_mask: bool,
    ) -> Dataset:
        roi_root = self._cfg.roi_dir if use_roi else None
        mask_root = self._cfg.mask_dir if use_mask else None
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
            pin_memory=True,
            drop_last=False,
        )

    def train_loader(self) -> DataLoader:
        dataset = self._build_dataset(
            root=self._cfg.train_dir,
            nominal_only=self._cfg.nominal_train_only,
            use_roi=True,
            use_mask=False,
        )
        return self._loader(dataset, shuffle=True)

    def val_loader(self) -> DataLoader | None:
        if self._cfg.val_dir is not None:
            dataset = self._build_dataset(
                root=self._cfg.val_dir,
                nominal_only=self._cfg.nominal_train_only,
                use_roi=True,
                use_mask=False,
            )
            return self._loader(dataset, shuffle=False)

        dataset = self._build_dataset(
            root=self._cfg.train_dir,
            nominal_only=self._cfg.nominal_train_only,
            use_roi=True,
            use_mask=False,
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
        )
        return self._loader(dataset, shuffle=False)
