"""Threshold calibration and binary metric computation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def calibrate_threshold_balanced_accuracy(labels: np.ndarray, scores: np.ndarray) -> float:
    """Select threshold maximizing balanced accuracy on calibration set."""
    thresholds = np.unique(scores)
    best_threshold = float(thresholds[0])
    best_bacc = -1.0

    for threshold in thresholds:
        preds = (scores >= threshold).astype(np.int64)
        tp = np.sum((preds == 1) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))

        tpr = tp / max(tp + fn, 1)
        tnr = tn / max(tn + fp, 1)
        bacc = 0.5 * (tpr + tnr)
        if bacc > best_bacc:
            best_bacc = bacc
            best_threshold = float(threshold)

    return best_threshold


def binary_classification_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute binary metrics from scalar anomaly scores."""
    preds = (scores >= threshold).astype(np.int64)

    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    balanced_accuracy = 0.5 * (tpr + tnr)

    auroc = float("nan")
    if len(np.unique(labels)) > 1:
        auroc = float(roc_auc_score(labels, scores))

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "tpr": float(tpr),
        "tnr": float(tnr),
        "balanced_accuracy": float(balanced_accuracy),
        "auroc": float(auroc),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }
