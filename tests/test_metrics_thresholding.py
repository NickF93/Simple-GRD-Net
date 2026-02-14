import math

import numpy as np

from grdnet.metrics.thresholding import (
    binary_classification_metrics,
    calibrate_threshold_balanced_accuracy,
)


def test_binary_metrics_include_extended_keys() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    scores = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    threshold = 0.5
    metrics = binary_classification_metrics(labels, scores, threshold)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["tp"] == 2.0
    assert metrics["tn"] == 2.0
    assert metrics["fp"] == 0.0
    assert metrics["fn"] == 0.0
    assert metrics["auroc"] == 1.0
    assert metrics["ap"] == 1.0
    assert metrics["average_precision"] == 1.0


def test_binary_metrics_nan_auc_and_ap_with_single_class() -> None:
    labels = np.asarray([0, 0, 0], dtype=np.int64)
    scores = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
    metrics = binary_classification_metrics(labels, scores, threshold=0.2)
    assert math.isnan(metrics["auroc"])
    assert math.isnan(metrics["ap"])
    assert math.isnan(metrics["average_precision"])


def test_calibrate_threshold_returns_member_of_score_set() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    scores = np.asarray([0.1, 0.4, 0.6, 0.9], dtype=np.float64)
    threshold = calibrate_threshold_balanced_accuracy(labels, scores)
    assert threshold in set(scores.tolist())
