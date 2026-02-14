"""Typed configuration schema for GRD-Net experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProfileConfig(BaseModel):
    """High-level paper profile selection."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    mode: Literal["grdnet_2023_full", "deepindustrial_sn_2026"]
    use_segmentator: bool = True

    @model_validator(mode="after")
    def validate_mode(self) -> "ProfileConfig":
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
    patch_size: int = Field(ge=16)
    patch_stride: int = Field(ge=1)
    nominal_train_only: bool = True

    @model_validator(mode="after")
    def validate_patch_shape(self) -> "DataConfig":
        if self.patch_size > self.image_size:
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
    run_acceptance_ratio: float = Field(default=0.7, gt=0.0, le=1.0)


class ReportingConfig(BaseModel):
    """Reporting and artifact configuration."""

    model_config = ConfigDict(extra="forbid")

    csv_metrics_filename: str = "metrics.csv"
    csv_predictions_filename: str = "predictions.csv"


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
