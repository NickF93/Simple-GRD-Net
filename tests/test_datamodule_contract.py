from pathlib import Path

import pytest
from PIL import Image

from grdnet.config.schema import DataConfig
from grdnet.data.datamodule import DataModule


def test_test_loader_allows_missing_mask_dir_for_good_samples(tmp_path: Path) -> None:
    test_good_dir = tmp_path / "test" / "good"
    test_good_dir.mkdir(parents=True, exist_ok=True)
    Image.new("L", (32, 32), color=0).save(test_good_dir / "000.png")

    cfg = DataConfig(
        train_dir=tmp_path / "train",
        test_dir=tmp_path / "test",
        image_size=32,
        channels=1,
        batch_size=1,
        num_workers=0,
        patch_size=(32, 32),
        patch_stride=(32, 32),
    )

    datamodule = DataModule(cfg)
    loader = datamodule.test_loader()
    assert loader is not None
    batch = next(iter(loader))
    assert int(batch["label"][0]) == 0
    assert float(batch["gt_mask"][0].sum().item()) == 0.0


def test_test_loader_without_mask_dir_defaults_anomalous_samples_to_good(
    tmp_path: Path,
) -> None:
    test_anomaly_dir = tmp_path / "test" / "crack"
    test_anomaly_dir.mkdir(parents=True, exist_ok=True)
    Image.new("L", (32, 32), color=0).save(test_anomaly_dir / "000.png")

    cfg = DataConfig(
        train_dir=tmp_path / "train",
        test_dir=tmp_path / "test",
        image_size=32,
        channels=1,
        batch_size=1,
        num_workers=0,
        patch_size=(32, 32),
        patch_stride=(32, 32),
    )

    datamodule = DataModule(cfg)
    loader = datamodule.test_loader()
    assert loader is not None
    batch = next(iter(loader))
    assert int(batch["label"][0]) == 0
    assert float(batch["gt_mask"][0].sum().item()) == 0.0


def test_mvtec_category_root_infers_ground_truth_mask_dir(tmp_path: Path) -> None:
    category_root = tmp_path / "leather"
    train_good = category_root / "train" / "good"
    test_good = category_root / "test" / "good"
    ground_truth = category_root / "ground_truth"
    train_good.mkdir(parents=True, exist_ok=True)
    test_good.mkdir(parents=True, exist_ok=True)
    ground_truth.mkdir(parents=True, exist_ok=True)

    Image.new("L", (32, 32), color=0).save(train_good / "000.png")
    Image.new("L", (32, 32), color=0).save(test_good / "000.png")

    cfg = DataConfig(
        train_dir=category_root,
        test_dir=category_root,
        image_size=32,
        channels=1,
        batch_size=1,
        num_workers=0,
        patch_size=(32, 32),
        patch_stride=(32, 32),
    )

    datamodule = DataModule(cfg)
    loader = datamodule.test_loader()
    assert loader is not None
    batch = next(iter(loader))
    assert int(batch["label"][0]) == 0


def test_mvtec_benchmark_root_combines_categories(tmp_path: Path) -> None:
    for category in ("carpet", "capsule"):
        train_good = tmp_path / category / "train" / "good"
        test_good = tmp_path / category / "test" / "good"
        ground_truth = tmp_path / category / "ground_truth"
        train_good.mkdir(parents=True, exist_ok=True)
        test_good.mkdir(parents=True, exist_ok=True)
        ground_truth.mkdir(parents=True, exist_ok=True)
        Image.new("L", (32, 32), color=0).save(train_good / "000.png")

    cfg = DataConfig(
        train_dir=tmp_path,
        image_size=32,
        channels=1,
        batch_size=1,
        num_workers=0,
        patch_size=(32, 32),
        patch_stride=(32, 32),
    )

    datamodule = DataModule(cfg)
    loader = datamodule.train_loader()
    assert len(loader.dataset) == 2


def test_val_loader_falls_back_to_train_split(tmp_path: Path) -> None:
    train_good = tmp_path / "train" / "good"
    train_good.mkdir(parents=True, exist_ok=True)
    Image.new("L", (32, 32), color=0).save(train_good / "000.png")
    Image.new("L", (32, 32), color=0).save(train_good / "001.png")

    cfg = DataConfig(
        train_dir=tmp_path / "train",
        image_size=32,
        channels=1,
        batch_size=1,
        num_workers=0,
        patch_size=(32, 32),
        patch_stride=(32, 32),
    )
    datamodule = DataModule(cfg)
    loader = datamodule.val_loader()
    assert loader is not None


def test_optional_test_and_calibration_loaders_return_none(tmp_path: Path) -> None:
    cfg = DataConfig(
        train_dir=tmp_path / "train",
        image_size=32,
        channels=1,
        batch_size=1,
        num_workers=0,
        patch_size=(32, 32),
        patch_stride=(32, 32),
    )
    datamodule = DataModule(cfg)
    assert datamodule.test_loader() is None
    assert datamodule.calibration_loader() is None


def test_test_loader_uses_roi_when_configured(tmp_path: Path) -> None:
    test_good_dir = tmp_path / "test" / "good"
    roi_good_dir = tmp_path / "roi" / "good"
    test_good_dir.mkdir(parents=True, exist_ok=True)
    roi_good_dir.mkdir(parents=True, exist_ok=True)
    Image.new("L", (32, 32), color=0).save(test_good_dir / "000.png")
    Image.new("L", (32, 32), color=0).save(roi_good_dir / "000.png")

    cfg = DataConfig(
        train_dir=tmp_path / "train",
        test_dir=tmp_path / "test",
        roi_dir=tmp_path / "roi",
        image_size=32,
        channels=1,
        batch_size=1,
        num_workers=0,
        patch_size=(32, 32),
        patch_stride=(32, 32),
    )

    datamodule = DataModule(cfg)
    loader = datamodule.test_loader()
    assert loader is not None
    batch = next(iter(loader))
    assert float(batch["roi_mask"][0].sum().item()) == 0.0


def test_calibration_loader_uses_roi_when_configured(tmp_path: Path) -> None:
    test_good_dir = tmp_path / "test" / "good"
    roi_good_dir = tmp_path / "roi" / "good"
    test_good_dir.mkdir(parents=True, exist_ok=True)
    roi_good_dir.mkdir(parents=True, exist_ok=True)
    Image.new("L", (32, 32), color=0).save(test_good_dir / "000.png")
    Image.new("L", (32, 32), color=0).save(roi_good_dir / "000.png")

    cfg = DataConfig(
        train_dir=tmp_path / "train",
        calibration_dir=tmp_path / "test",
        roi_dir=tmp_path / "roi",
        image_size=32,
        channels=1,
        batch_size=1,
        num_workers=0,
        patch_size=(32, 32),
        patch_stride=(32, 32),
    )

    datamodule = DataModule(cfg)
    loader = datamodule.calibration_loader()
    assert loader is not None
    batch = next(iter(loader))
    assert float(batch["roi_mask"][0].sum().item()) == 0.0


def test_missing_roi_path_logs_warning_and_falls_back_to_ones(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    test_good_dir = tmp_path / "test" / "good"
    test_good_dir.mkdir(parents=True, exist_ok=True)
    Image.new("L", (32, 32), color=0).save(test_good_dir / "000.png")

    cfg = DataConfig(
        train_dir=tmp_path / "train",
        test_dir=tmp_path / "test",
        roi_dir=tmp_path / "roi_missing",
        image_size=32,
        channels=1,
        batch_size=1,
        num_workers=0,
        patch_size=(32, 32),
        patch_stride=(32, 32),
    )

    datamodule = DataModule(cfg)
    with caplog.at_level("WARNING"):
        loader = datamodule.test_loader()
        assert loader is not None
        batch = next(iter(loader))

    assert float(batch["roi_mask"][0].sum().item()) > 0.0
    assert any(
        "roi_dir_not_found" in rec.message or "roi_root_not_found" in rec.message
        for rec in caplog.records
    )


def test_train_loader_does_not_warn_when_masks_are_disabled(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    train_good_dir = tmp_path / "train" / "good"
    train_good_dir.mkdir(parents=True, exist_ok=True)
    Image.new("L", (32, 32), color=0).save(train_good_dir / "000.png")

    cfg = DataConfig(
        train_dir=tmp_path / "train",
        image_size=32,
        channels=1,
        batch_size=1,
        num_workers=0,
        patch_size=(32, 32),
        patch_stride=(32, 32),
    )

    datamodule = DataModule(cfg)
    with caplog.at_level("WARNING"):
        loader = datamodule.train_loader()
        _ = next(iter(loader))

    assert not any("mask_root_not_set" in rec.message for rec in caplog.records)
