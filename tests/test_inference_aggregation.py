from pathlib import Path

import numpy as np
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


def test_evaluate_reports_image_level_metrics_from_patch_ratio() -> None:
    cfg = load_experiment_config(Path("configs/profiles/deepindustrial_sn_2026.yaml"))
    cfg.data.patch_size = (2, 2)
    cfg.data.patch_stride = (2, 2)
    cfg.inference.run_acceptance_ratio = 0.5

    backend = _InferBackendStub()
    engine = InferenceEngine(cfg=cfg, backend=backend, reporters=[])

    batch = {
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

    metrics = engine.evaluate([batch], threshold=0.5)
    assert metrics["image.accuracy"] == 1.0
    assert metrics["image.balanced_accuracy"] == 1.0

    rows = engine.infer([batch], threshold=0.5)
    assert all("image_prediction" in row for row in rows)
    assert all("anomalous_patch_ratio" in row for row in rows)


def test_calibrate_uses_patch_threshold_when_image_ratio_disabled(monkeypatch) -> None:
    cfg = load_experiment_config(Path("configs/profiles/deepindustrial_sn_2026.yaml"))
    cfg.data.patch_size = (2, 2)
    cfg.data.patch_stride = (2, 2)
    cfg.inference.run_acceptance_ratio = 0.0

    backend = _InferBackendStub()
    engine = InferenceEngine(cfg=cfg, backend=backend, reporters=[])

    batch = {
        "image": torch.zeros((2, 1, 4, 4), dtype=torch.float32),
        "gt_mask": torch.zeros((2, 1, 4, 4), dtype=torch.float32),
        "path": ["img_1.png", "img_2.png"],
    }

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
    patch_paths = np.asarray(["img1.png", "img1.png", "img2.png", "img2.png"])
    threshold = engine._calibrate_threshold_image_ratio(
        patch_labels=patch_labels,
        patch_scores=patch_scores,
        patch_paths=patch_paths,
    )
    assert threshold in {0.1, 0.2, 0.8, 0.9}
