import torch

from grdnet.config.schema import DataConfig
from grdnet.data.adapters.custom_mvtec_like import CustomMvtecLikeAdapter
from grdnet.data.adapters.mvtec import MvtecLikeAdapter
from grdnet.data.contracts import SampleItem


def test_custom_adapter_is_mvtec_adapter_alias(tmp_path) -> None:
    cfg = DataConfig(
        train_dir=tmp_path / "train",
        image_size=32,
        channels=1,
        batch_size=1,
        num_workers=0,
        patch_size=(32, 32),
        patch_stride=(32, 32),
    )
    adapter = CustomMvtecLikeAdapter(cfg)
    assert isinstance(adapter, MvtecLikeAdapter)


def test_sample_item_contract_fields() -> None:
    item = SampleItem(
        image=torch.zeros((1, 8, 8), dtype=torch.float32),
        label=0,
        path="image.png",
        roi_mask=torch.ones((1, 8, 8), dtype=torch.float32),
        gt_mask=torch.zeros((1, 8, 8), dtype=torch.float32),
    )
    assert item.path == "image.png"
    assert item.label == 0
