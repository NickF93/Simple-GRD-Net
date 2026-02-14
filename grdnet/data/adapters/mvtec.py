"""MVTec-style directory dataset adapter."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from grdnet.config.schema import DataConfig
from grdnet.core.exceptions import DatasetContractError

_ALLOWED_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _IndexedSample:
    image_path: Path
    class_name: str


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

    def __len__(self) -> int:
        return len(self._samples)

    def _load_image(self, path: Path) -> torch.Tensor:
        mode = "L" if self._cfg.channels == 1 else "RGB"
        with Image.open(path) as image:
            image = image.convert(mode)
            image = image.resize((self._cfg.image_size, self._cfg.image_size), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0

        if self._cfg.channels == 1:
            array = np.expand_dims(array, axis=-1)

        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return tensor

    def _load_mask(self, base_path: Path, root: Path | None, default_fill: float) -> torch.Tensor:
        if root is None:
            return torch.full((1, self._cfg.image_size, self._cfg.image_size), default_fill, dtype=torch.float32)

        rel = base_path.parent.name + "/" + base_path.name
        mask_path = root / rel
        if not mask_path.exists():
            LOGGER.debug("Mask missing at %s; using default_fill=%s", mask_path, default_fill)
            return torch.full(
                (1, self._cfg.image_size, self._cfg.image_size),
                default_fill,
                dtype=torch.float32,
            )

        with Image.open(mask_path) as image:
            image = image.convert("L")
            image = image.resize((self._cfg.image_size, self._cfg.image_size), Image.Resampling.NEAREST)
            array = np.asarray(image, dtype=np.float32) / 255.0

        mask = torch.from_numpy((array > 0.5).astype(np.float32)).unsqueeze(0)
        return mask

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        sample = self._samples[index]
        image = self._load_image(sample.image_path)
        label = 0 if sample.class_name == self._cfg.class_good_name else 1
        roi_mask = self._load_mask(sample.image_path, self._roi_root, default_fill=1.0)
        gt_mask = self._load_mask(sample.image_path, self._mask_root, default_fill=0.0)

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
                    samples.append(_IndexedSample(image_path=path, class_name=class_name))

        if not samples:
            raise DatasetContractError(
                f"No images found in {root}. Expected class subfolders with image files."
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
        samples = self._index(root=root, nominal_only=nominal_only)
        return MvtecLikeDataset(
            samples,
            data_cfg=self._cfg,
            roi_root=roi_root,
            mask_root=mask_root,
        )
