from pathlib import Path

import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from grdnet.backends.base import ModelBundle, OptimizerBundle, SchedulerBundle
from grdnet.core.exceptions import CheckpointError
from grdnet.training.checkpoints import load_checkpoint, save_checkpoint


class _BackendStub:
    def __init__(self, *, with_segmentator: bool) -> None:
        generator = nn.Linear(4, 4)
        discriminator = nn.Linear(4, 1)
        segmentator = nn.Linear(4, 4) if with_segmentator else None

        self.models = ModelBundle(
            generator=generator,
            discriminator=discriminator,
            segmentator=segmentator,
        )

        opt_generator = SGD(generator.parameters(), lr=0.1)
        opt_discriminator = SGD(discriminator.parameters(), lr=0.1)
        opt_segmentator = SGD(segmentator.parameters(), lr=0.1) if segmentator else None
        self.optimizers = OptimizerBundle(
            generator=opt_generator,
            discriminator=opt_discriminator,
            segmentator=opt_segmentator,
        )

        sch_generator = StepLR(opt_generator, step_size=1)
        sch_discriminator = StepLR(opt_discriminator, step_size=1)
        sch_segmentator = (
            StepLR(opt_segmentator, step_size=1) if opt_segmentator else None
        )
        self.schedulers = SchedulerBundle(
            generator=sch_generator,
            discriminator=sch_discriminator,
            segmentator=sch_segmentator,
        )

        self.device = torch.device("cpu")


def test_checkpoint_roundtrip_without_segmentator(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "epoch_0001.pt"
    backend_write = _BackendStub(with_segmentator=False)
    save_checkpoint(backend_write, checkpoint_path, epoch=1)

    backend_read = _BackendStub(with_segmentator=False)
    epoch = load_checkpoint(backend_read, checkpoint_path)
    assert epoch == 1

    for key, value in backend_write.models.generator.state_dict().items():
        assert torch.equal(value, backend_read.models.generator.state_dict()[key])


def test_checkpoint_missing_required_keys_raises(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "invalid.pt"
    torch.save({"epoch": 1}, checkpoint_path)

    backend = _BackendStub(with_segmentator=False)
    with pytest.raises(CheckpointError):
        _ = load_checkpoint(backend, checkpoint_path)


def test_checkpoint_segmentator_contract_mismatch_raises(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "segmentator_missing.pt"
    backend_write = _BackendStub(with_segmentator=False)
    save_checkpoint(backend_write, checkpoint_path, epoch=2)

    backend_read = _BackendStub(with_segmentator=True)
    with pytest.raises(CheckpointError):
        _ = load_checkpoint(backend_read, checkpoint_path)


def test_checkpoint_load_prefers_weights_only(tmp_path: Path, monkeypatch) -> None:
    checkpoint_path = tmp_path / "weights_only.pt"
    backend_write = _BackendStub(with_segmentator=False)
    save_checkpoint(backend_write, checkpoint_path, epoch=3)

    recorded: dict[str, object] = {}
    original_torch_load = torch.load

    def _spy_load(*args, **kwargs):
        recorded.update(kwargs)
        return original_torch_load(*args, **kwargs)

    monkeypatch.setattr("grdnet.training.checkpoints.torch.load", _spy_load)
    backend_read = _BackendStub(with_segmentator=False)
    _ = load_checkpoint(backend_read, checkpoint_path)
    assert recorded.get("weights_only") is True


def test_checkpoint_load_rejects_weights_only_incompatible_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint_path = tmp_path / "incompatible_weights_only.pt"
    backend_write = _BackendStub(with_segmentator=False)
    save_checkpoint(backend_write, checkpoint_path, epoch=4)
    original_torch_load = torch.load

    def _incompatible_load(*args, **kwargs):
        if "weights_only" in kwargs:
            raise TypeError("weights_only unsupported")
        return original_torch_load(*args, **kwargs)

    monkeypatch.setattr("grdnet.training.checkpoints.torch.load", _incompatible_load)
    backend_read = _BackendStub(with_segmentator=False)
    with pytest.raises(CheckpointError, match="Unable to read checkpoint"):
        _ = load_checkpoint(backend_read, checkpoint_path)


def test_checkpoint_not_found_raises(tmp_path: Path) -> None:
    backend = _BackendStub(with_segmentator=False)
    with pytest.raises(CheckpointError, match="Checkpoint not found"):
        _ = load_checkpoint(backend, tmp_path / "missing.pt")


def test_checkpoint_path_not_file_raises(tmp_path: Path) -> None:
    backend = _BackendStub(with_segmentator=False)
    directory = tmp_path / "dir"
    directory.mkdir()
    with pytest.raises(CheckpointError, match="not a file"):
        _ = load_checkpoint(backend, directory)


def test_checkpoint_invalid_epoch_value_raises(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "invalid_epoch.pt"
    torch.save(
        {
            "epoch": -1,
            "models": {"generator": {}, "discriminator": {}, "segmentator": None},
            "optimizers": {"generator": {}, "discriminator": {}, "segmentator": None},
            "schedulers": {"generator": {}, "discriminator": {}, "segmentator": None},
        },
        checkpoint_path,
    )
    backend = _BackendStub(with_segmentator=False)
    with pytest.raises(CheckpointError, match="epoch"):
        _ = load_checkpoint(backend, checkpoint_path)
