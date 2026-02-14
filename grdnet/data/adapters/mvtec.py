"""MVTec-style directory dataset adapter."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from grdnet.config.schema import DataConfig
from grdnet.core.exceptions import DatasetContractError

_ALLOWED_EXTENSIONS: tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _IndexedSample:
    image_path: Path
    class_name: str
    class_relative_path: Path


class MvtecLikeDataset(Dataset):
    """Dataset implementation for MVTec-like folder layouts."""

    def __init__(
        self,
        samples: list[_IndexedSample],
        *,
        data_cfg: DataConfig,
        roi_root: Path | None,
        mask_root: Path | None,
    ) -> None:
        if not samples:
            raise DatasetContractError("Dataset is empty after indexing.")
        self._samples = samples
        self._cfg = data_cfg
        self._roi_root = roi_root
        self._mask_root = mask_root
        self._logged_missing_roi_root = False
        self._logged_missing_roi_sample = False
        self._logged_missing_mask_root = False
        self._logged_missing_gt_mask = False

    def __len__(self) -> int:
        return len(self._samples)

    def _load_image(self, path: Path) -> torch.Tensor:
        mode = "L" if self._cfg.channels == 1 else "RGB"
        with Image.open(path) as image_handle:
            converted = image_handle.convert(mode)
            resized = converted.resize(
                (self._cfg.image_size, self._cfg.image_size),
                Image.Resampling.BILINEAR,
            )
            array = np.asarray(resized, dtype=np.float32) / 255.0

        if self._cfg.channels == 1:
            array = np.expand_dims(array, axis=-1)

        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return tensor

    def _load_binary_mask(self, mask_path: Path) -> torch.Tensor:
        with Image.open(mask_path) as image_handle:
            converted = image_handle.convert("L")
            resized = converted.resize(
                (self._cfg.image_size, self._cfg.image_size),
                Image.Resampling.NEAREST,
            )
            array = np.asarray(resized, dtype=np.float32) / 255.0

        return torch.from_numpy((array > 0.5).astype(np.float32)).unsqueeze(0)

    def _resolve_roi_mask_path(self, sample: _IndexedSample) -> Path | None:
        if self._roi_root is None:
            return None
        if not self._roi_root.exists():
            if not self._logged_missing_roi_root:
                LOGGER.warning(
                    "roi_root_not_found path=%s fallback=full_image_roi",
                    self._roi_root,
                )
                self._logged_missing_roi_root = True
            return None

        base = self._roi_root / sample.class_name / sample.class_relative_path
        candidates = (
            base,
            base.with_suffix(".png"),
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        if not self._logged_missing_roi_sample:
            LOGGER.warning(
                "roi_mask_not_found sample=%s roi_root=%s fallback=full_image_roi",
                sample.image_path,
                self._roi_root,
            )
            self._logged_missing_roi_sample = True
        return None

    def _resolve_gt_mask_path(self, sample: _IndexedSample) -> Path | None:
        if self._mask_root is None:
            return None

        base = self._mask_root / sample.class_name / sample.class_relative_path
        candidates = (
            base,
            base.with_name(f"{base.stem}_mask{base.suffix}"),
            base.with_suffix(".png"),
            base.with_name(f"{base.stem}_mask.png"),
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_roi_mask(self, sample: _IndexedSample) -> torch.Tensor:
        roi_path = self._resolve_roi_mask_path(sample)
        if roi_path is None:
            return torch.ones(
                (1, self._cfg.image_size, self._cfg.image_size),
                dtype=torch.float32,
            )
        return self._load_binary_mask(roi_path)

    def _load_gt_mask(self, sample: _IndexedSample) -> torch.Tensor:
        if self._mask_root is None:
            if not self._logged_missing_mask_root:
                LOGGER.warning(
                    "mask_root_not_set fallback=zero_mask_treat_as_good",
                )
                self._logged_missing_mask_root = True
            return torch.zeros(
                (1, self._cfg.image_size, self._cfg.image_size),
                dtype=torch.float32,
            )

        gt_mask_path = self._resolve_gt_mask_path(sample)
        if gt_mask_path is None:
            if not self._logged_missing_gt_mask:
                LOGGER.warning(
                    "gt_mask_not_found sample=%s fallback=zero_mask_treat_as_good",
                    sample.image_path,
                )
                self._logged_missing_gt_mask = True
            return torch.zeros(
                (1, self._cfg.image_size, self._cfg.image_size),
                dtype=torch.float32,
            )

        return self._load_binary_mask(gt_mask_path)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        sample = self._samples[index]
        image = self._load_image(sample.image_path)
        roi_mask = self._load_roi_mask(sample)
        gt_mask = self._load_gt_mask(sample)
        label = int((gt_mask > 0.5).any().item())

        return {
            "image": image,
            "label": int(label),
            "path": str(sample.image_path),
            "roi_mask": roi_mask,
            "gt_mask": gt_mask,
        }


class MvtecLikeAdapter:
    """Adapter for MVTec-AD and custom MVTec-like datasets."""

    def __init__(self, data_cfg: DataConfig) -> None:
        self._cfg = data_cfg

    def _index(self, root: Path, nominal_only: bool) -> list[_IndexedSample]:
        if not root.exists() or not root.is_dir():
            raise DatasetContractError(f"Dataset directory does not exist: {root}")

        samples: list[_IndexedSample] = []
        for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            class_name = class_dir.name
            if nominal_only and class_name != self._cfg.class_good_name:
                continue

            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in _ALLOWED_EXTENSIONS:
                    relative = path.relative_to(class_dir)
                    samples.append(
                        _IndexedSample(
                            image_path=path,
                            class_name=class_name,
                            class_relative_path=relative,
                        )
                    )

        if not samples:
            raise DatasetContractError(
                "No images found in "
                f"{root}. Expected class subfolders with image files."
            )
        return samples

    def build_dataset(
        self,
        root: Path,
        *,
        nominal_only: bool,
        roi_root: Path | None,
        mask_root: Path | None,
    ) -> Dataset:
        """Index one split root and build a dataset instance."""
        samples = self._index(root=root, nominal_only=nominal_only)
        return MvtecLikeDataset(
            samples,
            data_cfg=self._cfg,
            roi_root=roi_root,
            mask_root=mask_root,
        )
