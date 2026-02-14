"""Inference, calibration, and evaluation workflows."""

from __future__ import annotations

import itertools

import numpy as np
import torch
from torch.utils.data import DataLoader

from grdnet.backends.base import BackendStrategy
from grdnet.config.schema import ExperimentConfig
from grdnet.data.patches import extract_patches
from grdnet.metrics.thresholding import (
    binary_classification_metrics,
    calibrate_threshold_balanced_accuracy,
)
from grdnet.reporting.base import Reporter


class InferenceEngine:
    """Backend-agnostic inference and metric aggregation."""

    def __init__(
        self,
        *,
        cfg: ExperimentConfig,
        backend: BackendStrategy,
        reporters: list[Reporter],
    ) -> None:
        self.cfg = cfg
        self.backend = backend
        self.reporters = reporters

    def _patch_labels(self, gt_mask: torch.Tensor) -> np.ndarray:
        patches = extract_patches(
            gt_mask,
            patch_size=self.cfg.data.patch_size,
            stride=self.cfg.data.patch_stride,
        )
        labels = (patches.flatten(start_dim=1).amax(dim=1) > 0.5).to(dtype=torch.int64)
        return labels.cpu().numpy()

    def _expanded_paths(self, paths: list[str], n_patches_per_image: int) -> list[str]:
        return list(itertools.chain.from_iterable([[path] * n_patches_per_image for path in paths]))

    def _collect_scores(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, list[dict[str, str | float | int]]]:
        all_scores: list[float] = []
        all_labels: list[int] = []
        rows: list[dict[str, str | float | int]] = []

        for batch in loader:
            output = self.backend.infer_step(batch)
            patch_scores = output.patch_scores.detach().cpu().numpy()

            gt_mask = batch["gt_mask"]
            if not isinstance(gt_mask, torch.Tensor):
                raise TypeError("batch['gt_mask'] must be tensor")
            labels = self._patch_labels(gt_mask)

            images = batch["image"]
            if not isinstance(images, torch.Tensor):
                raise TypeError("batch['image'] must be tensor")
            n_images = images.shape[0]
            n_patches = patch_scores.shape[0]
            n_patches_per_image = max(n_patches // n_images, 1)

            paths_raw = batch["path"]
            if not isinstance(paths_raw, list):
                raise TypeError("batch['path'] must be list[str]")
            expanded_paths = self._expanded_paths(paths_raw, n_patches_per_image)

            for idx, (score, label) in enumerate(zip(patch_scores, labels, strict=True)):
                rows.append(
                    {
                        "path": expanded_paths[idx],
                        "patch_index": idx,
                        "score": float(score),
                        "label": int(label),
                    }
                )

            all_scores.extend([float(x) for x in patch_scores])
            all_labels.extend([int(x) for x in labels])

        return np.asarray(all_labels), np.asarray(all_scores), rows

    def calibrate(self, loader: DataLoader) -> float:
        labels, scores, _ = self._collect_scores(loader)
        threshold = calibrate_threshold_balanced_accuracy(labels, scores)
        metrics = binary_classification_metrics(labels, scores, threshold)
        for reporter in self.reporters:
            reporter.log_evaluation({"phase": "calibration", **metrics})
        return threshold

    def evaluate(self, loader: DataLoader, threshold: float) -> dict[str, float]:
        labels, scores, rows = self._collect_scores(loader)
        metrics = binary_classification_metrics(labels, scores, threshold)
        for reporter in self.reporters:
            reporter.log_evaluation(metrics)
            reporter.write_predictions(rows)
        return metrics

    def infer(self, loader: DataLoader, threshold: float) -> list[dict[str, str | float | int]]:
        labels, scores, rows = self._collect_scores(loader)
        _ = labels
        for row, score in zip(rows, scores, strict=True):
            row["prediction"] = int(score >= threshold)
        for reporter in self.reporters:
            reporter.write_predictions(rows)
        return rows
