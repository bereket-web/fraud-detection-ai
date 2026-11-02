"""Metrics and validation-only threshold selection for rare-event classification."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def select_threshold(
    y_true: np.ndarray, scores: np.ndarray, objective: str = "f1", recall_floor: float = 0.80
) -> float:
    """Choose an alert threshold on validation data only."""
    y = np.asarray(y_true, dtype=np.int8)
    values = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or values.ndim != 1 or len(y) != len(values) or len(y) == 0:
        raise ValueError("Labels and scores must be non-empty, aligned one-dimensional arrays.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Threshold selection requires both positive and negative validation examples.")
    precision, recall, thresholds = precision_recall_curve(y, values)
    if not len(thresholds):
        raise ValueError("A usable threshold could not be selected.")

    if objective == "f1":
        denominator = precision[:-1] + recall[:-1]
        f1 = np.divide(
            2 * precision[:-1] * recall[:-1],
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )
        best = np.flatnonzero(np.isclose(f1, np.max(f1)))
        return float(thresholds[best[-1]])
    if objective == "recall":
        eligible = np.flatnonzero(recall[:-1] >= recall_floor)
        if len(eligible):
            best_precision = np.max(precision[eligible])
            best = eligible[np.isclose(precision[eligible], best_precision)]
            return float(thresholds[best[-1]])
        # Use the lowest threshold if the validation data cannot reach the requested recall.
        return float(thresholds[0])
    raise ValueError("objective must be 'f1' or 'recall'.")


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    """Return ranking and alert metrics at a fixed threshold."""
    y = np.asarray(y_true, dtype=np.int8)
    values = np.asarray(scores, dtype=np.float64)
    predictions = (values >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result: dict[str, float | int] = {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "false_positive_rate": float(1 - specificity),
        "average_precision": float(average_precision_score(y, values)),
        "roc_auc": float(roc_auc_score(y, values)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "alerts": int(tp + fp),
        "actual_fraud": int(tp + fn),
        "transactions": int(len(y)),
        "prevalence": float(np.mean(y)),
    }
    return result
