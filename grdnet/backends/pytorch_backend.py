"""PyTorch backend implementation (full v1)."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable
from contextlib import nullcontext

import torch
from torch import nn
from torch.optim import AdamW, Optimizer

from grdnet.backends.base import (
    BackendStrategy,
    ModelBundle,
    OptimizerBundle,
    SchedulerBundle,
    StepOutput,
    TrainBatchPreview,
)
from grdnet.config.schema import ExperimentConfig
from grdnet.core.exceptions import ConfigurationError
from grdnet.data.patches import extract_patches
from grdnet.losses.pytorch_losses import GrdNetLossComputer
from grdnet.metrics.anomaly import anomaly_heatmap, anomaly_score_ssim_per_sample
from grdnet.models.pytorch.discriminator import Discriminator
from grdnet.models.pytorch.generator import GeneratorEDE
from grdnet.models.pytorch.segmentator import UNetSegmentator
from grdnet.training.perturbation import (
    apply_gaussian_noise,
    apply_geometry_augmentation,
    apply_perlin_perturbation,
)
from grdnet.training.schedulers import GammaCosineAnnealingWarmRestarts

LOGGER = logging.getLogger(__name__)


class PyTorchBackend(BackendStrategy):
    """Concrete backend using PyTorch for all computations."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__(cfg)
        self._device = self._resolve_device(cfg.backend.device)
        LOGGER.info("backend=pytorch selected_device=%s", self._device)
        self._amp_enabled = cfg.backend.mixed_precision
        if self._amp_enabled and self._device.type != "cuda":
            raise ConfigurationError(
                "backend.mixed_precision=true requires CUDA device. "
                "Set backend.device='auto' with CUDA available, "
                "or disable mixed precision."
            )
        if self._amp_enabled:
            self._grad_scaler: torch.amp.GradScaler | None = torch.amp.GradScaler(
                "cuda",
                enabled=True,
            )
        else:
            self._grad_scaler = None
        LOGGER.info(
            "backend=pytorch mixed_precision_effective=%s autocast_dtype=%s",
            self._amp_enabled,
            "float16" if self._amp_enabled else "none",
        )
        self.losses = GrdNetLossComputer(cfg)

        self.models = self.build_models()
        self.optimizers = self.build_optimizers(self.models)
        self.schedulers = self.build_schedulers(self.optimizers)

    @staticmethod
    def _resolve_device(raw: str) -> torch.device:
        if raw == "auto":
            if not torch.backends.cuda.is_built():
                return torch.device("cpu")
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=UserWarning)
                    is_available = torch.cuda.is_available()
            except Warning as exc:
                raise ConfigurationError(
                    "backend.device='auto' detected CUDA build, but CUDA "
                    f"initialization failed: {exc}. "
                    "Fix CUDA driver/runtime or set backend.device='cpu'."
                ) from exc
            return torch.device("cuda" if is_available else "cpu")

        device = torch.device(raw)
        if device.type == "cuda":
            if not torch.backends.cuda.is_built():
                raise ConfigurationError(
                    "backend.device='cuda' requested, but this PyTorch build "
                    "does not include CUDA support."
                )
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=UserWarning)
                    is_available = torch.cuda.is_available()
            except Warning as exc:
                raise ConfigurationError(
                    "backend.device='cuda' requested, but CUDA initialization "
                    f"failed: {exc}"
                ) from exc
            if not is_available:
                raise ConfigurationError(
                    "backend.device='cuda' requested, but no CUDA device is "
                    "available."
                )
        return device

    @staticmethod
    def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    @staticmethod
    def _reset_optimizer_step_flag(optimizer: Optimizer) -> None:
        """Reset per-step optimizer marker used by scheduler ordering checks."""
        optimizer._opt_called = False  # type: ignore[attr-defined]

    def _reset_optimizer_step_flags(self) -> None:
        self._reset_optimizer_step_flag(self.optimizers.generator)
        self._reset_optimizer_step_flag(self.optimizers.discriminator)
        if self.optimizers.segmentator is not None:
            self._reset_optimizer_step_flag(self.optimizers.segmentator)

    def _set_train_modes(
        self,
        *,
        generator: bool,
        discriminator: bool,
        segmentator: bool,
    ) -> None:
        """Set module train/eval states for one optimization phase."""
        self.models.generator.train(generator)
        self.models.discriminator.train(discriminator)
        if self.models.segmentator is not None:
            self.models.segmentator.train(segmentator)

    @property
    def device(self) -> torch.device:
        """Return backend runtime device."""
        return self._device

    def _autocast(self):
        if not self._amp_enabled:
            return nullcontext()
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        )

    def _backward_step(
        self,
        *,
        loss: torch.Tensor,
        optimizer: Optimizer,
        parameters: Iterable[torch.nn.Parameter] | None = None,
        max_grad_norm: float | None = None,
    ) -> None:
        if self._amp_enabled:
            if self._grad_scaler is None:
                raise RuntimeError("GradScaler must be initialized when AMP is enabled")
            self._grad_scaler.scale(loss).backward()
            if max_grad_norm is not None and parameters is not None:
                self._grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(parameters), max_grad_norm)
            self._grad_scaler.step(optimizer)
            self._grad_scaler.update()
            return

        loss.backward()
        if max_grad_norm is not None and parameters is not None:
            torch.nn.utils.clip_grad_norm_(list(parameters), max_grad_norm)
        optimizer.step()

    def build_models(self) -> ModelBundle:
        """Instantiate model modules for the selected profile."""
        patch_shape = self.cfg.data.patch_size
        generator = GeneratorEDE(
            in_channels=self.cfg.data.channels,
            base_features=self.cfg.model.base_features,
            stages=self.cfg.model.stages,
            latent_dim=self.cfg.model.latent_dim,
            dense_bottleneck=self.cfg.model.dense_bottleneck,
            image_shape=patch_shape,
            encoder_downsample_position=self.cfg.model.encoder_downsample_position,
            decoder_upsample_position=self.cfg.model.decoder_upsample_position,
        ).to(self.device)

        discriminator = Discriminator(
            in_channels=self.cfg.data.channels,
            base_features=self.cfg.model.base_features,
            stages=self.cfg.model.stages,
            image_shape=patch_shape,
            encoder_downsample_position=self.cfg.model.encoder_downsample_position,
        ).to(self.device)

        segmentator: nn.Module | None = None
        if self.cfg.profile.use_segmentator:
            segmentator = UNetSegmentator(
                in_channels=self.cfg.data.channels * 2,
                base_features=self.cfg.model.segmentator_base_features,
            ).to(self.device)

        return ModelBundle(
            generator=generator,
            discriminator=discriminator,
            segmentator=segmentator,
        )

    def build_optimizers(self, models: ModelBundle) -> OptimizerBundle:
        """Instantiate optimizers for all active model branches."""
        betas = self.cfg.optimizer.adam_betas
        generator = AdamW(
            models.generator.parameters(),
            lr=self.cfg.optimizer.lr_generator,
            betas=betas,
            weight_decay=self.cfg.optimizer.weight_decay,
        )
        discriminator = AdamW(
            models.discriminator.parameters(),
            lr=self.cfg.optimizer.lr_discriminator,
            betas=betas,
            weight_decay=self.cfg.optimizer.weight_decay,
        )

        segmentator = None
        if models.segmentator is not None:
            segmentator = AdamW(
                models.segmentator.parameters(),
                lr=self.cfg.optimizer.lr_segmentator,
                betas=betas,
                weight_decay=self.cfg.optimizer.weight_decay,
            )
        return OptimizerBundle(
            generator=generator,
            discriminator=discriminator,
            segmentator=segmentator,
        )

    def build_schedulers(self, optimizers: OptimizerBundle) -> SchedulerBundle:
        """Instantiate learning-rate schedulers for all active optimizers."""
        generator = GammaCosineAnnealingWarmRestarts(
            optimizers.generator,
            first_restart_steps=self.cfg.scheduler.first_restart_steps,
            restart_t_mult=self.cfg.scheduler.restart_t_mult,
            restart_gamma=self.cfg.scheduler.restart_gamma,
        )
        discriminator = GammaCosineAnnealingWarmRestarts(
            optimizers.discriminator,
            first_restart_steps=self.cfg.scheduler.first_restart_steps,
            restart_t_mult=self.cfg.scheduler.restart_t_mult,
            restart_gamma=self.cfg.scheduler.restart_gamma,
        )

        segmentator = None
        if optimizers.segmentator is not None:
            segmentator = GammaCosineAnnealingWarmRestarts(
                optimizers.segmentator,
                first_restart_steps=self.cfg.scheduler.first_restart_steps,
                restart_t_mult=self.cfg.scheduler.restart_t_mult,
                restart_gamma=self.cfg.scheduler.restart_gamma,
            )

        return SchedulerBundle(
            generator=generator,
            discriminator=discriminator,
            segmentator=segmentator,
        )

    def _prepare(
        self,
        batch: dict[str, torch.Tensor | int | str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = batch["image"]
        roi_mask = batch["roi_mask"]
        gt_mask = batch["gt_mask"]
        if not isinstance(image, torch.Tensor):
            raise TypeError("batch['image'] must be a torch.Tensor")
        if not isinstance(roi_mask, torch.Tensor):
            raise TypeError("batch['roi_mask'] must be a torch.Tensor")
        if not isinstance(gt_mask, torch.Tensor):
            raise TypeError("batch['gt_mask'] must be a torch.Tensor")

        image = image.to(self.device)
        roi_mask = roi_mask.to(self.device)
        gt_mask = gt_mask.to(self.device)

        image = extract_patches(
            image,
            patch_size=self.cfg.data.patch_size,
            stride=self.cfg.data.patch_stride,
        )
        roi_mask = extract_patches(
            roi_mask,
            patch_size=self.cfg.data.patch_size,
            stride=self.cfg.data.patch_stride,
        )
        gt_mask = extract_patches(
            gt_mask,
            patch_size=self.cfg.data.patch_size,
            stride=self.cfg.data.patch_stride,
        )

        roi_mask = (roi_mask > 0.5).to(dtype=torch.float32)
        gt_mask = (gt_mask > 0.5).to(dtype=torch.float32)
        return image, roi_mask, gt_mask

    def _common_forward(
        self,
        x: torch.Tensor,
        roi_mask: torch.Tensor,
        *,
        with_augmentation: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        x, roi_mask, x_noisy, noise, noise_mask, beta = self._build_noisy_inputs(
            x=x,
            roi_mask=roi_mask,
            with_augmentation=with_augmentation,
        )

        z, x_rebuilt, z_rebuilt = self.models.generator(x_noisy)

        # Eq. (6) inspired noise penalty used in simple variant paper.
        noise_pred = torch.abs(
            ((1.0 - beta) * noise_mask * x_rebuilt) - (noise_mask * x_noisy)
        )
        noise_target = beta * noise
        noise_loss = torch.nn.functional.mse_loss(noise_pred, noise_target)

        return x, roi_mask, noise_mask, z, x_rebuilt, z_rebuilt, noise_loss, x_noisy

    def _build_noisy_inputs(
        self,
        *,
        x: torch.Tensor,
        roi_mask: torch.Tensor,
        with_augmentation: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Create perturbed tuple for GAN phases."""
        if with_augmentation:
            x, roi_mask = apply_geometry_augmentation(
                x,
                roi_mask,
                self.cfg.augmentation,
            )

        x_noisy, noise, noise_mask, beta = apply_perlin_perturbation(
            x,
            self.cfg.augmentation,
        )
        x_noisy = apply_gaussian_noise(
            x_noisy,
            self.cfg.augmentation.gaussian_noise_std_max,
        )
        return x, roi_mask, x_noisy, noise, noise_mask, beta

    def train_step(self, batch: dict[str, torch.Tensor | int | str]) -> StepOutput:
        """Execute one training step over GAN and optional segmentator branches."""
        self._reset_optimizer_step_flags()
        x, roi_mask, _ = self._prepare(batch)

        with self._autocast():
            x, roi_mask, x_noisy, noise, noise_mask, beta = self._build_noisy_inputs(
                x=x,
                roi_mask=roi_mask,
                with_augmentation=True,
            )

        # Discriminator phase: train D only, use detached fake built with no graph.
        self._set_train_modes(
            generator=False,
            discriminator=True,
            segmentator=False,
        )
        with self._autocast(), torch.no_grad():
            _, x_rebuilt_disc, _ = self.models.generator(x_noisy)
        self.optimizers.discriminator.zero_grad(set_to_none=True)
        self._set_requires_grad(self.models.discriminator, True)
        with self._autocast():
            _, pred_real_logits = self.models.discriminator(x)
            _, pred_fake_logits = self.models.discriminator(x_rebuilt_disc)
            loss_disc, loss_disc_scalar = self.losses.discriminator_total(
                pred_real_logits,
                pred_fake_logits,
            )
        self._backward_step(loss=loss_disc, optimizer=self.optimizers.discriminator)
        del x_rebuilt_disc

        # Generator phase: train G only, keep D in eval mode to freeze BN stats.
        self._set_train_modes(
            generator=True,
            discriminator=False,
            segmentator=False,
        )
        self.optimizers.generator.zero_grad(set_to_none=True)
        self._set_requires_grad(self.models.discriminator, False)
        with self._autocast():
            z, x_rebuilt, z_rebuilt = self.models.generator(x_noisy)

            noise_pred = torch.abs(
                ((1.0 - beta) * noise_mask * x_rebuilt) - (noise_mask * x_noisy)
            )
            noise_target = beta * noise
            noise_loss = torch.nn.functional.mse_loss(noise_pred, noise_target)

            with torch.no_grad():
                feat_real_gen, _ = self.models.discriminator(x)
            feat_fake_gen, _ = self.models.discriminator(x_rebuilt)
            loss_gen, stats_gen = self.losses.generator_total(
                x=x,
                x_rebuilt=x_rebuilt,
                z=z,
                z_rebuilt=z_rebuilt,
                feat_real=feat_real_gen,
                feat_fake=feat_fake_gen,
                noise_loss=noise_loss,
            )
        self._backward_step(
            loss=loss_gen,
            optimizer=self.optimizers.generator,
            parameters=self.models.generator.parameters(),
            max_grad_norm=self.cfg.training.max_grad_norm,
        )
        self._set_requires_grad(self.models.discriminator, True)

        seg_map: torch.Tensor | None = None
        seg_loss_scalar = 0.0
        if (
            self.models.segmentator is not None
            and self.optimizers.segmentator is not None
        ):
            self._set_train_modes(
                generator=False,
                discriminator=False,
                segmentator=True,
            )
            self.optimizers.segmentator.zero_grad(set_to_none=True)
            with self._autocast():
                seg_map = self.models.segmentator(
                    torch.cat([x_noisy, x_rebuilt.detach()], dim=1)
                )
                loss_seg, seg_loss_scalar = self.losses.segmentator_total(
                    pred_mask=seg_map,
                    gt_mask=noise_mask,
                    roi_mask=roi_mask,
                )
            self._backward_step(
                loss=loss_seg,
                optimizer=self.optimizers.segmentator,
            )

        with torch.no_grad():
            patch_scores = anomaly_score_ssim_per_sample(x, x_rebuilt)
            heatmap = anomaly_heatmap(x, x_rebuilt)

        stats = {
            **stats_gen,
            "loss.discriminator": float(loss_disc_scalar),
            "loss.segmentator": float(seg_loss_scalar),
        }
        preview: TrainBatchPreview | None = None
        if self.cfg.reporting.train_batch_preview.enabled:
            preview = TrainBatchPreview(
                x=x.detach(),
                x_noisy=x_noisy.detach(),
                noise_mask=noise_mask.detach(),
                x_rebuilt=x_rebuilt.detach(),
            )
        return StepOutput(
            stats=stats,
            x_rebuilt=x_rebuilt.detach(),
            patch_scores=patch_scores.detach(),
            heatmap=heatmap.detach(),
            seg_map=None if seg_map is None else seg_map.detach(),
            train_batch_preview=preview,
        )

    def eval_step(self, batch: dict[str, torch.Tensor | int | str]) -> StepOutput:
        """Execute one validation step without gradient updates."""
        self.models.generator.eval()
        self.models.discriminator.eval()
        if self.models.segmentator is not None:
            self.models.segmentator.eval()

        with torch.no_grad():
            x, roi_mask, _ = self._prepare(batch)
            with self._autocast():
                (
                    x,
                    roi_mask,
                    noise_mask,
                    z,
                    x_rebuilt,
                    z_rebuilt,
                    noise_loss,
                    x_noisy,
                ) = self._common_forward(x, roi_mask, with_augmentation=False)

                feat_real, pred_real_logits = self.models.discriminator(x)
                feat_fake, pred_fake_logits = self.models.discriminator(x_rebuilt)
                loss_gen, stats_gen = self.losses.generator_total(
                    x=x,
                    x_rebuilt=x_rebuilt,
                    z=z,
                    z_rebuilt=z_rebuilt,
                    feat_real=feat_real,
                    feat_fake=feat_fake,
                    noise_loss=noise_loss,
                )
                loss_disc, loss_disc_scalar = self.losses.discriminator_total(
                    pred_real_logits,
                    pred_fake_logits,
                )

                seg_map: torch.Tensor | None = None
                seg_loss_scalar = 0.0
                if self.models.segmentator is not None:
                    seg_map = self.models.segmentator(
                        torch.cat([x_noisy, x_rebuilt], dim=1)
                    )
                    _, seg_loss_scalar = self.losses.segmentator_total(
                        pred_mask=seg_map,
                        gt_mask=noise_mask,
                        roi_mask=roi_mask,
                    )

            patch_scores = anomaly_score_ssim_per_sample(x, x_rebuilt)
            heatmap = anomaly_heatmap(x, x_rebuilt)

            stats = {
                **stats_gen,
                "loss.discriminator": float(loss_disc_scalar),
                "loss.segmentator": float(seg_loss_scalar),
                "loss.total_eval": float(
                    loss_gen.detach().cpu().item() + loss_disc.detach().cpu().item()
                ),
            }
            return StepOutput(
                stats=stats,
                x_rebuilt=x_rebuilt,
                patch_scores=patch_scores,
                heatmap=heatmap,
                seg_map=seg_map,
            )

    def infer_step(self, batch: dict[str, torch.Tensor | int | str]) -> StepOutput:
        """Execute one inference step and return per-patch scores."""
        self.models.generator.eval()
        if self.models.segmentator is not None:
            self.models.segmentator.eval()

        with torch.no_grad():
            x, roi_mask, _ = self._prepare(batch)
            with self._autocast():
                z, x_rebuilt, _ = self.models.generator(x)
            del z

            seg_map: torch.Tensor | None = None
            if (
                self.models.segmentator is not None
                and self.cfg.profile.mode == "grdnet_2023_full"
            ):
                with self._autocast():
                    seg_map = self.models.segmentator(torch.cat([x, x_rebuilt], dim=1))

            scoring_strategy = self.cfg.inference.scoring_strategy
            if scoring_strategy == "profile_default":
                scoring_strategy = (
                    "segmentator_roi_max"
                    if self.cfg.profile.mode == "grdnet_2023_full"
                    else "ssim"
                )

            if scoring_strategy == "segmentator_roi_max" and seg_map is None:
                raise RuntimeError(
                    "segmentator_roi_max scoring requires segmentator "
                    "enabled and available"
                )

            if scoring_strategy == "segmentator_roi_max" and seg_map is not None:
                roi_intersection = seg_map * roi_mask
                patch_scores = roi_intersection.flatten(start_dim=1).amax(dim=1)
                heatmap = roi_intersection
            else:
                patch_scores = anomaly_score_ssim_per_sample(x, x_rebuilt)
                heatmap = anomaly_heatmap(x, x_rebuilt)

            stats = {
                "score.mean": float(patch_scores.mean().detach().cpu().item()),
                "score.max": float(patch_scores.max().detach().cpu().item()),
            }
            return StepOutput(
                stats=stats,
                x_rebuilt=x_rebuilt,
                patch_scores=patch_scores,
                heatmap=heatmap,
                seg_map=seg_map,
            )
