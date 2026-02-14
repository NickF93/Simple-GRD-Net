import torch

from grdnet.metrics.anomaly import (
    anomaly_heatmap,
    anomaly_score_l1,
    anomaly_score_ssim_per_sample,
)


def test_anomaly_heatmap_is_normalized_per_sample() -> None:
    x = torch.zeros((2, 1, 4, 4), dtype=torch.float32)
    x_rebuilt = torch.zeros_like(x)
    x_rebuilt[0, 0, 0, 0] = 1.0
    x_rebuilt[1, 0, 1, 1] = 2.0

    heatmap = anomaly_heatmap(x, x_rebuilt)
    assert heatmap.shape == (2, 1, 4, 4)
    assert float(heatmap.min().item()) == 0.0
    assert float(heatmap.max().item()) == 1.0


def test_anomaly_score_l1_matches_manual_mean_abs() -> None:
    x = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]], dtype=torch.float32)
    y = torch.tensor([[[[1.0, 1.0], [2.0, 1.0]]]], dtype=torch.float32)
    score = anomaly_score_l1(x, y)
    assert score.shape == (1,)
    assert torch.isclose(score[0], torch.tensor(0.75))


def test_anomaly_score_ssim_is_zero_for_identical_inputs() -> None:
    x = torch.rand((3, 1, 8, 8), dtype=torch.float32)
    score = anomaly_score_ssim_per_sample(x, x)
    assert torch.allclose(score, torch.zeros_like(score), atol=1e-5)
