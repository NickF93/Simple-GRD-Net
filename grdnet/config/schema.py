"""Typed configuration schema for GRD-Net experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ProfileConfig(BaseModel):
    """High-level paper profile selection."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    mode: Literal["grdnet_2023_full", "deepindustrial_sn_2026"]
    use_segmentator: bool = True

    @model_validator(mode="after")
    def validate_mode(self) -> ProfileConfig:
        """Ensure profile-mode feature combinations remain valid."""
        if self.mode == "deepindustrial_sn_2026" and self.use_segmentator:
            raise ValueError(
                "deepindustrial_sn_2026 must disable segmentator. "
                "Set profile.use_segmentator to false."
            )
        return self


class BackendConfig(BaseModel):
    """Backend strategy configuration."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["pytorch", "tensorflow_scaffold"]
    device: str = "auto"
    mixed_precision: bool = False


class DataConfig(BaseModel):
    """Dataset and loader contracts."""

    model_config = ConfigDict(extra="forbid")

    train_dir: Path
    val_dir: Path | None = None
    test_dir: Path | None = None
    calibration_dir: Path | None = None
    roi_dir: Path | None = None
    mask_dir: Path | None = None
    class_good_name: str = "good"
    image_size: int = Field(ge=32)
    channels: Literal[1, 3]
    batch_size: int = Field(ge=1)
    num_workers: int = Field(ge=0)
    patch_size: tuple[int, int]
    patch_stride: tuple[int, int]
    nominal_train_only: bool = True

    @field_validator("patch_size", "patch_stride", mode="before")
    @classmethod
    def parse_pair(cls, value: Any) -> tuple[int, int]:
        """Parse square shorthand into explicit (height, width) pairs."""
        if isinstance(value, int):
            return (value, value)
        if isinstance(value, tuple | list) and len(value) == 2:
            first = int(value[0])
            second = int(value[1])
            return (first, second)
        raise TypeError("patch_size/patch_stride must be int or pair [h, w]")

    @model_validator(mode="after")
    def validate_patch_shape(self) -> DataConfig:
        """Validate patch dimensions and stride constraints."""
        patch_h, patch_w = self.patch_size
        stride_h, stride_w = self.patch_stride
        if patch_h < 16 or patch_w < 16:
            raise ValueError("data.patch_size dimensions must be >= 16")
        if stride_h < 1 or stride_w < 1:
            raise ValueError("data.patch_stride dimensions must be >= 1")
        if patch_h > self.image_size or patch_w > self.image_size:
            raise ValueError("data.patch_size cannot exceed data.image_size")
        return self


class AugmentationConfig(BaseModel):
    """Training-time augmentation and perturbation controls."""

    model_config = ConfigDict(extra="forbid")

    perlin_probability: float = Field(ge=0.0, le=1.0)
    perlin_min_area: int = Field(ge=0)
    rotation_degrees: tuple[float, float] = (-22.5, 22.5)
    horizontal_flip_probability: float = Field(ge=0.0, le=1.0)
    vertical_flip_probability: float = Field(ge=0.0, le=1.0)
    gaussian_noise_std_max: float = Field(ge=0.0)


class ModelConfig(BaseModel):
    """Neural architecture hyperparameters."""

    model_config = ConfigDict(extra="forbid")

    base_features: int = Field(ge=8)
    latent_dim: int = Field(ge=8)
    stages: tuple[int, ...] = (2, 2, 2, 2)
    encoder_downsample_position: Literal["first", "last"] = "last"
    decoder_upsample_position: Literal["first", "last"] = "last"
    dense_bottleneck: bool = False
    segmentator_base_features: int = Field(ge=8)


class LossConfig(BaseModel):
    """Paper-aligned loss weighting values."""

    model_config = ConfigDict(extra="forbid")

    wa: float = Field(gt=0.0)
    wb: float = Field(gt=0.0)
    w1: float = Field(gt=0.0)
    w2: float = Field(gt=0.0)
    w3: float = Field(gt=0.0)
    w4: float = Field(ge=0.0)
    focal_gamma: float = Field(gt=0.0)
    focal_alpha: float = Field(gt=0.0, lt=1.0)
    contextual_base: Literal["l1", "huber"] = "l1"
    use_noise_regularization: bool = False


class OptimizerConfig(BaseModel):
    """Optimizer and scheduler controls."""

    model_config = ConfigDict(extra="forbid")

    lr_generator: float = Field(gt=0.0)
    lr_discriminator: float = Field(gt=0.0)
    lr_segmentator: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    adam_betas: tuple[float, float] = (0.5, 0.999)


class SchedulerConfig(BaseModel):
    """Cosine restart scheduler controls."""

    model_config = ConfigDict(extra="forbid")

    first_restart_steps: int = Field(ge=1)
    restart_t_mult: float = Field(ge=1.0)
    restart_gamma: float = Field(gt=0.0)
    step_unit: Literal["epoch", "step"] = "epoch"


class TrainingConfig(BaseModel):
    """Training loop controls."""

    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(ge=1)
    log_interval: int = Field(ge=1)
    eval_interval: int = Field(ge=1)
    checkpoint_dir: Path
    output_dir: Path
    max_grad_norm: float | None = Field(default=None, gt=0.0)


class InferenceConfig(BaseModel):
    """Inference and aggregation controls."""

    model_config = ConfigDict(extra="forbid")

    anomaly_threshold: float | None = Field(default=None, ge=0.0)
    run_acceptance_ratio: float = Field(default=0.7, ge=0.0, le=1.0)
    scoring_strategy: Literal[
        "profile_default",
        "ssim",
        "segmentator_roi_max",
    ] = "profile_default"


class TrainBatchPreviewConfig(BaseModel):
    """Training preview artifact controls."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    every_n_epochs: int = Field(default=1, ge=1)
    step_index: int = Field(default=1, ge=1)
    max_images: int = Field(default=8, ge=1)
    subdir: str = Field(default="train_batch_previews", min_length=1)


class ReportingConfig(BaseModel):
    """Reporting and artifact configuration."""

    model_config = ConfigDict(extra="forbid")

    csv_metrics_filename: str = "metrics.csv"
    csv_predictions_filename: str = "predictions.csv"
    train_batch_preview: TrainBatchPreviewConfig = Field(
        default_factory=TrainBatchPreviewConfig
    )


class SystemConfig(BaseModel):
    """Global system and reproducibility controls."""

    model_config = ConfigDict(extra="forbid")

    seed: int = Field(ge=0)
    deterministic: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class ExperimentConfig(BaseModel):
    """Root configuration object for any command."""

    model_config = ConfigDict(extra="forbid")

    profile: ProfileConfig
    backend: BackendConfig
    system: SystemConfig
    data: DataConfig
    augmentation: AugmentationConfig
    model: ModelConfig
    losses: LossConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    training: TrainingConfig
    inference: InferenceConfig
    reporting: ReportingConfig
