from pathlib import Path

import numpy as np
import pytest
import torch

from grdnet.backends.base import StepOutput
from grdnet.config.loader import load_experiment_config
from grdnet.inference.engine import InferenceEngine


class _InferBackendStub:
    def infer_step(self, batch: dict[str, torch.Tensor | list[str]]) -> StepOutput:
        _ = batch
        # 2 images, 4 patches each (patch_size=2, stride=2 over 4x4)
        scores = torch.tensor(
            [0.1, 0.2, 0.1, 0.2, 0.9, 0.8, 0.1, 0.1],
            dtype=torch.float32,
        )
        return StepOutput(
            stats={},
            x_rebuilt=torch.zeros((8, 1, 2, 2)),
            patch_scores=scores,
            heatmap=torch.zeros((8, 1, 2, 2)),
            seg_map=None,
        )


class _ReporterStub:
    def __init__(self) -> None:
        self.evaluations: list[dict[str, float | str]] = []
        self.prediction_rows: list[list[dict[str, str | float | int]]] = []

    def log_epoch(self, *, epoch: int, split: str, metrics: dict[str, float]) -> None:
        _ = epoch, split, metrics

    def log_evaluation(self, metrics: dict[str, float | str]) -> None:
        self.evaluations.append(metrics)

    def write_predictions(self, rows: list[dict[str, str | float | int]]) -> None:
        self.prediction_rows.append(rows)


def _cfg() -> object:
    cfg = load_experiment_config(Path("configs/profiles/deepindustrial_sn_2026.yaml"))
    cfg.data.patch_size = (2, 2)
    cfg.data.patch_stride = (2, 2)
    return cfg


def _batch() -> dict[str, torch.Tensor | list[str]]:
    return {
        "image": torch.zeros((2, 1, 4, 4), dtype=torch.float32),
        "gt_mask": torch.tensor(
            [
                [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]],
                [[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]],
            ],
            dtype=torch.float32,
        ),
        "path": ["img_1.png", "img_2.png"],
    }


def test_evaluate_reports_image_level_metrics_from_patch_ratio() -> None:
    cfg = _cfg()
    cfg.inference.run_acceptance_ratio = 0.5

    backend = _InferBackendStub()
    engine = InferenceEngine(cfg=cfg, backend=backend, reporters=[])

    batch = _batch()

    metrics = engine.evaluate([batch], threshold=0.5)
    assert metrics["image.accuracy"] == 1.0
    assert metrics["image.balanced_accuracy"] == 1.0

    prediction_rows = engine.infer([batch], threshold=0.5)
    assert prediction_rows == 8


def test_calibrate_uses_patch_threshold_when_image_ratio_disabled(monkeypatch) -> None:
    cfg = _cfg()
    cfg.inference.run_acceptance_ratio = 0.0

    backend = _InferBackendStub()
    engine = InferenceEngine(cfg=cfg, backend=backend, reporters=[])

    batch = _batch()
    batch["gt_mask"] = torch.zeros((2, 1, 4, 4), dtype=torch.float32)

    monkeypatch.setattr(
        "grdnet.inference.engine.calibrate_threshold_balanced_accuracy",
        lambda labels, scores: 0.123,
    )
    threshold = engine.calibrate([batch])
    assert threshold == 0.123


def test_image_ratio_helpers_compute_expected_values() -> None:
    cfg = load_experiment_config(Path("configs/profiles/deepindustrial_sn_2026.yaml"))
    engine = InferenceEngine(cfg=cfg, backend=_InferBackendStub(), reporters=[])

    bacc = engine._binary_balanced_accuracy(
        labels=np.asarray([0, 0, 1, 1], dtype=np.int64),
        preds=np.asarray([0, 1, 1, 1], dtype=np.int64),
    )
    assert bacc == 0.75

    patch_labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    patch_scores = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    threshold = engine._calibrate_threshold_image_ratio(
        patch_labels=patch_labels,
        patch_scores=patch_scores,
        n_patches_per_image=2,
    )
    assert threshold in {0.1, 0.2, 0.8, 0.9}


def test_validate_batch_patch_layout_raises_on_invalid_shapes() -> None:
    cfg = _cfg()
    engine = InferenceEngine(cfg=cfg, backend=_InferBackendStub(), reporters=[])

    with pytest.raises(ValueError, match="cannot be evenly mapped"):
        engine._validate_batch_patch_layout(
            images=torch.zeros((2, 1, 4, 4), dtype=torch.float32),
            patch_scores=np.asarray([0.1, 0.2, 0.3], dtype=np.float64),
            expected_patches_per_image=None,
        )

    with pytest.raises(ValueError, match="Patch layout changed across batches"):
        engine._validate_batch_patch_layout(
            images=torch.zeros((2, 1, 4, 4), dtype=torch.float32),
            patch_scores=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
            expected_patches_per_image=3,
        )


def test_collect_patch_arrays_raises_on_empty_loader() -> None:
    cfg = _cfg()
    engine = InferenceEngine(cfg=cfg, backend=_InferBackendStub(), reporters=[])
    with pytest.raises(ValueError, match="produced no batches"):
        engine._collect_patch_arrays([])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("gt_mask", "not-a-tensor", "batch\\['gt_mask'\\] must be tensor"),
        ("image", "not-a-tensor", "batch\\['image'\\] must be tensor"),
    ],
)
def test_collect_patch_arrays_raises_on_invalid_batch_types(
    field: str,
    value: str,
    message: str,
) -> None:
    cfg = _cfg()
    engine = InferenceEngine(cfg=cfg, backend=_InferBackendStub(), reporters=[])
    batch = _batch()
    batch[field] = value

    with pytest.raises(TypeError, match=message):
        engine._collect_patch_arrays([batch])


def test_candidate_thresholds_downsamples_large_vectors() -> None:
    cfg = _cfg()
    engine = InferenceEngine(cfg=cfg, backend=_InferBackendStub(), reporters=[])
    patch_scores = np.linspace(0.0, 1.0, num=5000, dtype=np.float64)

    thresholds = engine._candidate_thresholds(patch_scores)
    assert thresholds.size <= 2048
    assert thresholds[0] == 0.0
    assert thresholds[-1] == 1.0


def test_calibrate_image_ratio_path_logs_metrics() -> None:
    cfg = _cfg()
    cfg.inference.run_acceptance_ratio = 0.5
    reporter = _ReporterStub()
    engine = InferenceEngine(cfg=cfg, backend=_InferBackendStub(), reporters=[reporter])

    threshold = engine.calibrate([_batch()])
    assert isinstance(threshold, float)
    assert reporter.evaluations
    assert reporter.evaluations[0]["phase"] == "calibration"


def test_evaluate_streams_prediction_rows_and_logs_metrics() -> None:
    cfg = _cfg()
    cfg.inference.run_acceptance_ratio = 0.5
    reporter = _ReporterStub()
    engine = InferenceEngine(cfg=cfg, backend=_InferBackendStub(), reporters=[reporter])

    _ = engine.evaluate([_batch()], threshold=0.5)
    assert reporter.prediction_rows
    assert reporter.evaluations


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("gt_mask", "not-a-tensor", "batch\\['gt_mask'\\] must be tensor"),
        ("image", "not-a-tensor", "batch\\['image'\\] must be tensor"),
        ("path", "not-a-list", "batch\\['path'\\] must be list\\[str\\]"),
        ("path", ["img_1.png"], "length must match batch image count"),
    ],
)
def test_evaluate_raises_on_invalid_batch_contract(
    field: str,
    value: str | list[str],
    message: str,
) -> None:
    cfg = _cfg()
    engine = InferenceEngine(cfg=cfg, backend=_InferBackendStub(), reporters=[])
    batch = _batch()
    batch[field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        engine.evaluate([batch], threshold=0.5)


def test_evaluate_raises_on_empty_loader() -> None:
    cfg = _cfg()
    engine = InferenceEngine(cfg=cfg, backend=_InferBackendStub(), reporters=[])
    with pytest.raises(ValueError, match="produced no batches"):
        engine.evaluate([], threshold=0.5)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("image", "not-a-tensor", "batch\\['image'\\] must be tensor"),
        ("path", "not-a-list", "batch\\['path'\\] must be list\\[str\\]"),
        ("path", ["img_1.png"], "length must match batch image count"),
    ],
)
def test_infer_raises_on_invalid_batch_contract(
    field: str,
    value: str | list[str],
    message: str,
) -> None:
    cfg = _cfg()
    engine = InferenceEngine(cfg=cfg, backend=_InferBackendStub(), reporters=[])
    batch = _batch()
    batch[field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        engine.infer([batch], threshold=0.5)


def test_infer_raises_on_empty_loader() -> None:
    cfg = _cfg()
    engine = InferenceEngine(cfg=cfg, backend=_InferBackendStub(), reporters=[])
    with pytest.raises(ValueError, match="produced no batches"):
        engine.infer([], threshold=0.5)


def test_emit_prediction_rows_supports_empty_chunks() -> None:
    cfg = _cfg()
    reporter = _ReporterStub()
    engine = InferenceEngine(cfg=cfg, backend=_InferBackendStub(), reporters=[reporter])

    engine._emit_prediction_rows([])
    assert reporter.prediction_rows == []
