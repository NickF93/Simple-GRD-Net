"""Inference, calibration, and evaluation workflows."""

from __future__ import annotations

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

    @staticmethod
    def _prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
        return {f"{prefix}.{key}": value for key, value in metrics.items()}

    @staticmethod
    def _binary_balanced_accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
        tp = np.sum((preds == 1) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        tpr = tp / max(tp + fn, 1)
        tnr = tn / max(tn + fp, 1)
        return float(0.5 * (tpr + tnr))

    @staticmethod
    def _group_paths(paths: np.ndarray) -> tuple[list[str], dict[str, np.ndarray]]:
        groups: dict[str, list[int]] = {}
        ordered_paths: list[str] = []
        for idx, path in enumerate(paths.tolist()):
            if path not in groups:
                groups[path] = []
                ordered_paths.append(path)
            groups[path].append(idx)

        np_groups = {
            path: np.asarray(indices, dtype=np.int64)
            for path, indices in groups.items()
        }
        return ordered_paths, np_groups

    def _aggregate_image_level(
        self,
        *,
        patch_paths: np.ndarray,
        patch_labels: np.ndarray,
        patch_scores: np.ndarray,
        patch_preds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, str | float | int]]]:
        ordered_paths, groups = self._group_paths(patch_paths)

        image_labels: list[int] = []
        image_scores: list[float] = []
        image_preds: list[int] = []
        image_rows: list[dict[str, str | float | int]] = []

        for path in ordered_paths:
            indices = groups[path]
            labels = patch_labels[indices]
            scores = patch_scores[indices]
            preds = patch_preds[indices]

            anomaly_ratio = float(np.mean(preds))
            image_label = int(np.max(labels))
            image_pred = int(anomaly_ratio >= self.cfg.inference.run_acceptance_ratio)
            image_score = anomaly_ratio

            image_labels.append(image_label)
            image_scores.append(image_score)
            image_preds.append(image_pred)
            image_rows.append(
                {
                    "path": path,
                    "patch_count": int(indices.size),
                    "image_label": image_label,
                    "anomalous_patch_ratio": anomaly_ratio,
                    "image_prediction": image_pred,
                }
            )

        return (
            np.asarray(image_labels, dtype=np.int64),
            np.asarray(image_scores, dtype=np.float64),
            np.asarray(image_preds, dtype=np.int64),
            image_rows,
        )

    def _collect_scores(
        self,
        loader: DataLoader,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, str | float | int]]]:
        all_scores: list[float] = []
        all_labels: list[int] = []
        all_paths: list[str] = []
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
            if n_images <= 0 or n_patches % n_images != 0:
                raise ValueError(
                    "Patch tensor cannot be evenly mapped back to batch images"
                )
            n_patches_per_image = n_patches // n_images

            paths_raw = batch["path"]
            if not isinstance(paths_raw, list):
                raise TypeError("batch['path'] must be list[str]")
            if len(paths_raw) != n_images:
                raise ValueError("batch['path'] length must match batch image count")

            for image_idx, image_path in enumerate(paths_raw):
                start = image_idx * n_patches_per_image
                end = start + n_patches_per_image
                for patch_idx, flat_idx in enumerate(range(start, end), start=0):
                    rows.append(
                        {
                            "path": image_path,
                            "patch_index": int(patch_idx),
                            "score": float(patch_scores[flat_idx]),
                            "label": int(labels[flat_idx]),
                        }
                    )
                    all_paths.append(image_path)
                    all_scores.append(float(patch_scores[flat_idx]))
                    all_labels.append(int(labels[flat_idx]))

        return (
            np.asarray(all_labels, dtype=np.int64),
            np.asarray(all_scores, dtype=np.float64),
            np.asarray(all_paths),
            rows,
        )

    def _calibrate_threshold_image_ratio(
        self,
        *,
        patch_labels: np.ndarray,
        patch_scores: np.ndarray,
        patch_paths: np.ndarray,
    ) -> float:
        thresholds = np.unique(patch_scores)
        best_threshold = float(thresholds[0])
        best_bacc = -1.0

        for threshold in thresholds:
            patch_preds = (patch_scores >= threshold).astype(np.int64)
            image_labels, _, image_preds, _ = self._aggregate_image_level(
                patch_paths=patch_paths,
                patch_labels=patch_labels,
                patch_scores=patch_scores,
                patch_preds=patch_preds,
            )
            bacc = self._binary_balanced_accuracy(image_labels, image_preds)
            if bacc > best_bacc:
                best_bacc = bacc
                best_threshold = float(threshold)

        return best_threshold

    def calibrate(self, loader: DataLoader) -> float:
        patch_labels, patch_scores, patch_paths, _ = self._collect_scores(loader)

        if self.cfg.inference.run_acceptance_ratio > 0.0:
            threshold = self._calibrate_threshold_image_ratio(
                patch_labels=patch_labels,
                patch_scores=patch_scores,
                patch_paths=patch_paths,
            )
        else:
            threshold = calibrate_threshold_balanced_accuracy(
                patch_labels,
                patch_scores,
            )

        patch_metrics = binary_classification_metrics(
            patch_labels,
            patch_scores,
            threshold,
        )
        patch_preds = (patch_scores >= threshold).astype(np.int64)
        image_labels, image_scores, _, _ = self._aggregate_image_level(
            patch_paths=patch_paths,
            patch_labels=patch_labels,
            patch_scores=patch_scores,
            patch_preds=patch_preds,
        )
        image_metrics = binary_classification_metrics(
            image_labels,
            image_scores,
            self.cfg.inference.run_acceptance_ratio,
        )

        metrics = {
            "phase": "calibration",
            **self._prefix_metrics(patch_metrics, "patch"),
            **self._prefix_metrics(image_metrics, "image"),
        }
        for reporter in self.reporters:
            reporter.log_evaluation(metrics)
        return threshold

    def evaluate(self, loader: DataLoader, threshold: float) -> dict[str, float]:
        patch_labels, patch_scores, patch_paths, rows = self._collect_scores(loader)

        patch_metrics = binary_classification_metrics(
            patch_labels,
            patch_scores,
            threshold,
        )
        patch_preds = (patch_scores >= threshold).astype(np.int64)
        image_labels, image_scores, _, image_rows = self._aggregate_image_level(
            patch_paths=patch_paths,
            patch_labels=patch_labels,
            patch_scores=patch_scores,
            patch_preds=patch_preds,
        )
        image_metrics = binary_classification_metrics(
            image_labels,
            image_scores,
            self.cfg.inference.run_acceptance_ratio,
        )

        image_pred_by_path = {
            str(row["path"]): int(row["image_prediction"])
            for row in image_rows
        }
        anomaly_ratio_by_path = {
            str(row["path"]): float(row["anomalous_patch_ratio"])
            for row in image_rows
        }
        for row, pred in zip(rows, patch_preds, strict=True):
            path = str(row["path"])
            row["patch_prediction"] = int(pred)
            row["image_prediction"] = image_pred_by_path[path]
            row["anomalous_patch_ratio"] = anomaly_ratio_by_path[path]

        metrics = {
            **self._prefix_metrics(patch_metrics, "patch"),
            **self._prefix_metrics(image_metrics, "image"),
        }
        for reporter in self.reporters:
            reporter.log_evaluation(metrics)
            reporter.write_predictions(rows)
        return metrics

    def infer(
        self,
        loader: DataLoader,
        threshold: float,
    ) -> list[dict[str, str | float | int]]:
        patch_labels, patch_scores, patch_paths, rows = self._collect_scores(loader)

        patch_preds = (patch_scores >= threshold).astype(np.int64)
        _, _, _, image_rows = self._aggregate_image_level(
            patch_paths=patch_paths,
            patch_labels=patch_labels,
            patch_scores=patch_scores,
            patch_preds=patch_preds,
        )
        image_pred_by_path = {
            str(row["path"]): int(row["image_prediction"])
            for row in image_rows
        }
        anomaly_ratio_by_path = {
            str(row["path"]): float(row["anomalous_patch_ratio"])
            for row in image_rows
        }

        for row, pred in zip(rows, patch_preds, strict=True):
            path = str(row["path"])
            row["patch_prediction"] = int(pred)
            row["image_prediction"] = image_pred_by_path[path]
            row["anomalous_patch_ratio"] = anomaly_ratio_by_path[path]

        for reporter in self.reporters:
            reporter.write_predictions(rows)
        return rows
