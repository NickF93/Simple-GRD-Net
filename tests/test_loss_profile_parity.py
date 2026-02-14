import torch
from torch.nn import functional

from grdnet.config.loader import load_experiment_config
from grdnet.losses.pytorch_losses import GrdNetLossComputer


def _toy_tensors() -> tuple[torch.Tensor, ...]:
    x = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]]], dtype=torch.float32)
    x_rebuilt = torch.tensor([[[[0.2, 0.1], [0.35, 0.45]]]], dtype=torch.float32)
    z = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    z_rebuilt = torch.tensor([[0.15, 0.25, 0.2]], dtype=torch.float32)
    feat_real = torch.tensor([[0.3, 0.4]], dtype=torch.float32)
    feat_fake = torch.tensor([[0.1, 0.6]], dtype=torch.float32)
    noise_loss = torch.tensor(0.5, dtype=torch.float32)
    return x, x_rebuilt, z, z_rebuilt, feat_real, feat_fake, noise_loss


def test_grdnet_profile_uses_l1_contextual_and_disables_noise_term() -> None:
    cfg = load_experiment_config("configs/profiles/grdnet_2023_full.yaml")
    loss = GrdNetLossComputer(cfg)
    x, x_rebuilt, z, z_rebuilt, feat_real, feat_fake, noise_loss = _toy_tensors()

    contextual = loss.contextual(x, x_rebuilt)
    expected_contextual = (
        cfg.losses.wa * functional.l1_loss(x_rebuilt, x)
        + cfg.losses.wb * loss.ssim(x_rebuilt, x)
    )
    assert torch.isclose(contextual, expected_contextual)

    total, _ = loss.generator_total(
        x=x,
        x_rebuilt=x_rebuilt,
        z=z,
        z_rebuilt=z_rebuilt,
        feat_real=feat_real,
        feat_fake=feat_fake,
        noise_loss=noise_loss,
    )
    adversarial = functional.mse_loss(feat_fake, feat_real)
    encoder = functional.l1_loss(z, z_rebuilt)
    expected_total = (
        cfg.losses.w1 * adversarial
        + cfg.losses.w2 * expected_contextual
        + cfg.losses.w3 * encoder
    )
    assert torch.isclose(total, expected_total)


def test_deepindustrial_profile_uses_huber_contextual_and_noise_term() -> None:
    cfg = load_experiment_config("configs/profiles/deepindustrial_sn_2026.yaml")
    loss = GrdNetLossComputer(cfg)
    x, x_rebuilt, z, z_rebuilt, feat_real, feat_fake, noise_loss = _toy_tensors()

    contextual = loss.contextual(x, x_rebuilt)
    expected_contextual = (
        cfg.losses.wa * functional.huber_loss(x_rebuilt, x, delta=1.0)
        + cfg.losses.wb * loss.ssim(x_rebuilt, x)
    )
    assert torch.isclose(contextual, expected_contextual)

    total, _ = loss.generator_total(
        x=x,
        x_rebuilt=x_rebuilt,
        z=z,
        z_rebuilt=z_rebuilt,
        feat_real=feat_real,
        feat_fake=feat_fake,
        noise_loss=noise_loss,
    )
    adversarial = functional.mse_loss(feat_fake, feat_real)
    encoder = functional.l1_loss(z, z_rebuilt)
    expected_total = (
        cfg.losses.w1 * adversarial
        + cfg.losses.w2 * expected_contextual
        + cfg.losses.w3 * encoder
        + cfg.losses.w4 * noise_loss
    )
    assert torch.isclose(total, expected_total)


def test_discriminator_loss_uses_logits_formulation() -> None:
    cfg = load_experiment_config("configs/profiles/deepindustrial_sn_2026.yaml")
    loss = GrdNetLossComputer(cfg)
    assert isinstance(loss.bce_logits, torch.nn.BCEWithLogitsLoss)

    pred_real_logits = torch.tensor([[6.0], [-2.0]], dtype=torch.float32)
    pred_fake_logits = torch.tensor([[-4.0], [3.0]], dtype=torch.float32)

    total, _ = loss.discriminator_total(
        pred_real_logits=pred_real_logits,
        pred_fake_logits=pred_fake_logits,
    )
    expected = 0.5 * (
        functional.binary_cross_entropy_with_logits(
            pred_real_logits,
            torch.ones_like(pred_real_logits),
        )
        + functional.binary_cross_entropy_with_logits(
            pred_fake_logits,
            torch.zeros_like(pred_fake_logits),
        )
    )
    assert torch.isclose(total, expected)
