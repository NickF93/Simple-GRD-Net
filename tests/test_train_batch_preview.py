from pathlib import Path

import pytest
import torch

from grdnet.backends.base import TrainBatchPreview
from grdnet.reporting.train_batch_preview import TrainBatchPreviewWriter


def _preview(batch: int = 4) -> TrainBatchPreview:
    return TrainBatchPreview(
        x=torch.rand((batch, 3, 16, 16), dtype=torch.float32),
        x_noisy=torch.rand((batch, 3, 16, 16), dtype=torch.float32),
        noise_mask=torch.rand((batch, 1, 16, 16), dtype=torch.float32),
        x_rebuilt=torch.rand((batch, 3, 16, 16), dtype=torch.float32),
    )


def test_train_batch_preview_writer_saves_composite(tmp_path: Path) -> None:
    writer = TrainBatchPreviewWriter(
        output_dir=tmp_path,
        subdir="batch_previews",
        max_images=3,
    )
    path = writer.write(epoch=2, step=1, preview=_preview(batch=5))
    assert path == tmp_path / "batch_previews" / "epoch_0002_step_0001.png"
    assert path.exists()
    assert path.stat().st_size > 0


def test_train_batch_preview_writer_rejects_invalid_channels(tmp_path: Path) -> None:
    writer = TrainBatchPreviewWriter(
        output_dir=tmp_path,
        subdir="batch_previews",
        max_images=3,
    )
    preview = _preview(batch=2)
    preview = TrainBatchPreview(
        x=preview.x,
        x_noisy=preview.x_noisy,
        noise_mask=torch.rand((2, 2, 16, 16), dtype=torch.float32),
        x_rebuilt=preview.x_rebuilt,
    )
    with pytest.raises(ValueError, match="noise_mask must have 1 or 3 channels"):
        _ = writer.write(epoch=1, step=1, preview=preview)


def test_train_batch_preview_writer_rejects_non_finite_values(tmp_path: Path) -> None:
    writer = TrainBatchPreviewWriter(
        output_dir=tmp_path,
        subdir="batch_previews",
        max_images=3,
    )
    preview = _preview(batch=2)
    preview = TrainBatchPreview(
        x=preview.x,
        x_noisy=preview.x_noisy,
        noise_mask=preview.noise_mask,
        x_rebuilt=torch.full((2, 3, 16, 16), float("nan"), dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="x_rebuilt contains non-finite values"):
        _ = writer.write(epoch=1, step=1, preview=preview)
