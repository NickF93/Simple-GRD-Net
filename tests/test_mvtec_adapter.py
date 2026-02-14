from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from grdnet.config.schema import DataConfig
from grdnet.data.adapters.mvtec import MvtecLikeAdapter


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)


def _data_cfg(tmp_path: Path) -> DataConfig:
    return DataConfig(
        train_dir=tmp_path / "train",
        image_size=32,
        channels=1,
        batch_size=1,
        num_workers=0,
        patch_size=(32, 32),
        patch_stride=(32, 32),
    )


def test_good_sample_allows_missing_mask_and_defaults_to_zeros(tmp_path: Path) -> None:
    test_root = tmp_path / "test"
    mask_root = tmp_path / "ground_truth"
    _write_png(test_root / "good" / "000.png", np.zeros((32, 32), dtype=np.uint8))

    adapter = MvtecLikeAdapter(_data_cfg(tmp_path))
    dataset = adapter.build_dataset(
        root=test_root,
        nominal_only=False,
        roi_root=None,
        mask_root=mask_root,
    )
    sample = dataset[0]
    assert int(sample["label"]) == 0
    assert float(sample["gt_mask"].sum().item()) == 0.0


def test_good_sample_missing_mask_does_not_warn(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    test_root = tmp_path / "test"
    mask_root = tmp_path / "ground_truth"
    _write_png(test_root / "good" / "000.png", np.zeros((32, 32), dtype=np.uint8))

    adapter = MvtecLikeAdapter(_data_cfg(tmp_path))
    dataset = adapter.build_dataset(
        root=test_root,
        nominal_only=False,
        roi_root=None,
        mask_root=mask_root,
    )
    with caplog.at_level("WARNING"):
        sample = dataset[0]

    assert int(sample["label"]) == 0
    assert float(sample["gt_mask"].sum().item()) == 0.0
    assert not any("gt_mask_not_found" in rec.message for rec in caplog.records)


def test_anomalous_sample_missing_mask_defaults_to_good(tmp_path: Path) -> None:
    test_root = tmp_path / "test"
    mask_root = tmp_path / "ground_truth"
    _write_png(test_root / "crack" / "000.png", np.zeros((32, 32), dtype=np.uint8))

    adapter = MvtecLikeAdapter(_data_cfg(tmp_path))
    dataset = adapter.build_dataset(
        root=test_root,
        nominal_only=False,
        roi_root=None,
        mask_root=mask_root,
    )
    sample = dataset[0]
    assert int(sample["label"]) == 0
    assert float(sample["gt_mask"].sum().item()) == 0.0


def test_mvtec_mask_suffix_and_nested_relative_paths_are_supported(
    tmp_path: Path,
) -> None:
    test_root = tmp_path / "test"
    mask_root = tmp_path / "ground_truth"
    _write_png(
        test_root / "scratch" / "nested" / "001.png",
        np.zeros((32, 32), dtype=np.uint8),
    )
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    _write_png(
        mask_root / "scratch" / "nested" / "001_mask.png",
        mask,
    )

    adapter = MvtecLikeAdapter(_data_cfg(tmp_path))
    dataset = adapter.build_dataset(
        root=test_root,
        nominal_only=False,
        roi_root=None,
        mask_root=mask_root,
    )
    sample = dataset[0]
    assert int(sample["label"]) == 1
    assert float(sample["gt_mask"].sum().item()) > 0.0
