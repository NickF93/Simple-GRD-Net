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
    def _validate_batch_patch_layout(
        *,
        images: torch.Tensor,
        patch_scores: np.ndarray,
        expected_patches_per_image: int | None,
    ) -> tuple[int, int]:
        """Validate and return `(n_images, n_patches_per_image)`."""
        n_images = images.shape[0]
        n_patches = patch_scores.shape[0]
        if n_images <= 0 or n_patches % n_images != 0:
            raise ValueError(
                "Patch tensor cannot be evenly mapped back to batch images"
            )
        n_patches_per_image = n_patches // n_images
        if (
            expected_patches_per_image is not None
            and expected_patches_per_image != n_patches_per_image
        ):
            raise ValueError(
                "Patch layout changed across batches. "
                "Check data.patch_size/patch_stride and batch tensor consistency."
            )
        return n_images, n_patches_per_image

    def _collect_patch_arrays(
        self,
        loader: DataLoader,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Collect patch labels/scores without path-level buffering."""
        patch_label_chunks: list[np.ndarray] = []
        patch_score_chunks: list[np.ndarray] = []
        n_patches_per_image: int | None = None

        for batch in loader:
            output = self.backend.infer_step(batch)
            patch_scores = output.patch_scores.detach().cpu().numpy().astype(np.float64)

            gt_mask = batch["gt_mask"]
            if not isinstance(gt_mask, torch.Tensor):
                raise TypeError("batch['gt_mask'] must be tensor")
            labels = self._patch_labels(gt_mask).astype(np.int64)

            images = batch["image"]
            if not isinstance(images, torch.Tensor):
                raise TypeError("batch['image'] must be tensor")
            _, n_patches_per_image = self._validate_batch_patch_layout(
                images=images,
                patch_scores=patch_scores,
                expected_patches_per_image=n_patches_per_image,
            )

            patch_label_chunks.append(labels)
            patch_score_chunks.append(patch_scores)

        if not patch_label_chunks:
            raise ValueError("Inference loader produced no batches")
        if n_patches_per_image is None:
            raise ValueError("Unable to resolve patch layout from inference loader")

        patch_labels = np.concatenate(patch_label_chunks, axis=0)
        patch_scores = np.concatenate(patch_score_chunks, axis=0)
        return patch_labels, patch_scores, n_patches_per_image

    def _emit_prediction_rows(
        self,
        rows: list[dict[str, str | float | int]],
    ) -> None:
        """Write one bounded prediction chunk to all reporters."""
        if not rows:
            return
        for reporter in self.reporters:
            reporter.write_predictions(rows)

    @staticmethod
    def _image_labels_from_patch_labels(
        patch_labels: np.ndarray,
        n_patches_per_image: int,
    ) -> np.ndarray:
        """Collapse patch labels into one image label by max pooling."""
        return (
            patch_labels.reshape(-1, n_patches_per_image)
            .max(axis=1)
            .astype(np.int64, copy=False)
        )

    @staticmethod
    def _image_scores_from_patch_preds(
        patch_preds: np.ndarray,
        n_patches_per_image: int,
    ) -> np.ndarray:
        """Collapse patch predictions into image anomaly ratio scores."""
        return patch_preds.reshape(-1, n_patches_per_image).mean(
            axis=1,
            dtype=np.float64,
        )

    @staticmethod
    def _candidate_thresholds(patch_scores: np.ndarray) -> np.ndarray:
        """Build bounded threshold candidates for image-ratio calibration."""
        thresholds = np.unique(patch_scores)
        if thresholds.size <= 2048:
            return thresholds
        quantiles = np.linspace(0.0, 1.0, num=2048, dtype=np.float64)
        sampled = np.quantile(patch_scores, quantiles, method="linear")
        return np.unique(sampled)

    def _calibrate_threshold_image_ratio(
        self,
        *,
        patch_labels: np.ndarray,
        patch_scores: np.ndarray,
        n_patches_per_image: int,
    ) -> float:
        thresholds = self._candidate_thresholds(patch_scores)
        best_threshold = float(thresholds[0])
        best_bacc = -1.0
        image_labels = self._image_labels_from_patch_labels(
            patch_labels,
            n_patches_per_image,
        )
        patch_score_matrix = patch_scores.reshape(-1, n_patches_per_image)

        for threshold in thresholds:
            image_scores = (patch_score_matrix >= threshold).mean(
                axis=1,
                dtype=np.float64,
            )
            image_preds = (
                image_scores >= self.cfg.inference.run_acceptance_ratio
            ).astype(np.int64)
            bacc = self._binary_balanced_accuracy(image_labels, image_preds)
            if bacc > best_bacc:
                best_bacc = bacc
                best_threshold = float(threshold)

        return best_threshold

    def calibrate(self, loader: DataLoader) -> float:
        """Calibrate patch threshold, then report patch and image metrics."""
        patch_labels, patch_scores, n_patches_per_image = self._collect_patch_arrays(
            loader
        )

        if self.cfg.inference.run_acceptance_ratio > 0.0:
            threshold = self._calibrate_threshold_image_ratio(
                patch_labels=patch_labels,
                patch_scores=patch_scores,
                n_patches_per_image=n_patches_per_image,
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
        image_labels = self._image_labels_from_patch_labels(
            patch_labels,
            n_patches_per_image,
        )
        image_scores = self._image_scores_from_patch_preds(
            patch_preds,
            n_patches_per_image,
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
        """Run evaluation with bounded-memory prediction streaming."""
        patch_label_chunks: list[np.ndarray] = []
        patch_score_chunks: list[np.ndarray] = []
        image_label_chunks: list[np.ndarray] = []
        image_score_chunks: list[np.ndarray] = []
        n_patches_per_image: int | None = None

        for batch in loader:
            output = self.backend.infer_step(batch)
            patch_scores = output.patch_scores.detach().cpu().numpy().astype(np.float64)

            gt_mask = batch["gt_mask"]
            if not isinstance(gt_mask, torch.Tensor):
                raise TypeError("batch['gt_mask'] must be tensor")
            patch_labels = self._patch_labels(gt_mask).astype(np.int64)

            images = batch["image"]
            if not isinstance(images, torch.Tensor):
                raise TypeError("batch['image'] must be tensor")
            n_images, n_patches_per_image = self._validate_batch_patch_layout(
                images=images,
                patch_scores=patch_scores,
                expected_patches_per_image=n_patches_per_image,
            )

            paths_raw = batch["path"]
            if not isinstance(paths_raw, list):
                raise TypeError("batch['path'] must be list[str]")
            if len(paths_raw) != n_images:
                raise ValueError("batch['path'] length must match batch image count")

            patch_preds = (patch_scores >= threshold).astype(np.int64)
            patch_pred_matrix = patch_preds.reshape(n_images, n_patches_per_image)
            patch_label_matrix = patch_labels.reshape(n_images, n_patches_per_image)
            patch_score_matrix = patch_scores.reshape(n_images, n_patches_per_image)
            image_labels = patch_label_matrix.max(axis=1).astype(np.int64)
            image_scores = patch_pred_matrix.mean(axis=1, dtype=np.float64)
            image_preds = (
                image_scores >= self.cfg.inference.run_acceptance_ratio
            ).astype(np.int64)

            chunk_rows: list[dict[str, str | float | int]] = []
            for image_idx, image_path in enumerate(paths_raw):
                anomaly_ratio = float(image_scores[image_idx])
                image_pred = int(image_preds[image_idx])
                image_label = int(image_labels[image_idx])
                for patch_idx in range(n_patches_per_image):
                    chunk_rows.append(
                        {
                            "path": image_path,
                            "patch_index": int(patch_idx),
                            "score": float(patch_score_matrix[image_idx, patch_idx]),
                            "label": int(patch_label_matrix[image_idx, patch_idx]),
                            "patch_prediction": int(
                                patch_pred_matrix[image_idx, patch_idx]
                            ),
                            "image_label": image_label,
                            "image_prediction": image_pred,
                            "anomalous_patch_ratio": anomaly_ratio,
                        }
                    )
            self._emit_prediction_rows(chunk_rows)

            patch_label_chunks.append(patch_labels)
            patch_score_chunks.append(patch_scores)
            image_label_chunks.append(image_labels)
            image_score_chunks.append(image_scores)

        if n_patches_per_image is None:
            raise ValueError("Inference loader produced no batches")

        patch_labels = np.concatenate(patch_label_chunks, axis=0)
        patch_scores = np.concatenate(patch_score_chunks, axis=0)
        image_labels = np.concatenate(image_label_chunks, axis=0)
        image_scores = np.concatenate(image_score_chunks, axis=0)

        patch_metrics = binary_classification_metrics(
            patch_labels,
            patch_scores,
            threshold,
        )
        image_metrics = binary_classification_metrics(
            image_labels,
            image_scores,
            self.cfg.inference.run_acceptance_ratio,
        )

        metrics = {
            **self._prefix_metrics(patch_metrics, "patch"),
            **self._prefix_metrics(image_metrics, "image"),
        }
        for reporter in self.reporters:
            reporter.log_evaluation(metrics)
        return metrics

    def infer(
        self,
        loader: DataLoader,
        threshold: float,
    ) -> int:
        """Run inference and stream predictions, returning written-row count."""
        n_patches_per_image: int | None = None
        prediction_count = 0

        for batch in loader:
            output = self.backend.infer_step(batch)
            patch_scores = output.patch_scores.detach().cpu().numpy().astype(np.float64)

            images = batch["image"]
            if not isinstance(images, torch.Tensor):
                raise TypeError("batch['image'] must be tensor")
            n_images, n_patches_per_image = self._validate_batch_patch_layout(
                images=images,
                patch_scores=patch_scores,
                expected_patches_per_image=n_patches_per_image,
            )

            paths_raw = batch["path"]
            if not isinstance(paths_raw, list):
                raise TypeError("batch['path'] must be list[str]")
            if len(paths_raw) != n_images:
                raise ValueError("batch['path'] length must match batch image count")

            patch_preds = (patch_scores >= threshold).astype(np.int64)
            patch_pred_matrix = patch_preds.reshape(n_images, n_patches_per_image)
            patch_score_matrix = patch_scores.reshape(n_images, n_patches_per_image)
            image_scores = patch_pred_matrix.mean(axis=1, dtype=np.float64)
            image_preds = (
                image_scores >= self.cfg.inference.run_acceptance_ratio
            ).astype(np.int64)

            chunk_rows: list[dict[str, str | float | int]] = []
            for image_idx, image_path in enumerate(paths_raw):
                anomaly_ratio = float(image_scores[image_idx])
                image_pred = int(image_preds[image_idx])
                for patch_idx in range(n_patches_per_image):
                    chunk_rows.append(
                        {
                            "path": image_path,
                            "patch_index": int(patch_idx),
                            "score": float(patch_score_matrix[image_idx, patch_idx]),
                            "patch_prediction": int(
                                patch_pred_matrix[image_idx, patch_idx]
                            ),
                            "image_prediction": image_pred,
                            "anomalous_patch_ratio": anomaly_ratio,
                        }
                    )

            prediction_count += len(chunk_rows)
            self._emit_prediction_rows(chunk_rows)

        if n_patches_per_image is None:
            raise ValueError("Inference loader produced no batches")
        return prediction_count
