from pathlib import Path

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
