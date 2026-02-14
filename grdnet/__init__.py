"""Official GRD-Net research framework.

This package provides a configurable implementation of:
- GRD-Net (2023): full generative-reconstructive-discriminative model with ROI-aware
  segmentation supervision.
- DeepIndustrial-SN (2026): runtime-optimized profile that uses the generative
  branch for patch-wise anomaly scoring and heatmap localization.
"""

from grdnet.config.loader import load_experiment_config

__all__ = ["load_experiment_config"]
__version__ = "1.0.0"
