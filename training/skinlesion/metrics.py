"""Evaluation metrics (pure functions over label arrays)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from skinlesion.config import CLASSES, CLASS_KEYS, HAM_CLASSES

DANGER_CATEGORIES = frozenset({"malignant", "precancerous"})


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> float:
    """Mean per-class recall over the given labels (the ISIC 2018 Task 3 metric)."""
    recalls = [np.mean(y_pred[y_true == c] == c) for c in labels if np.any(y_true == c)]
    return float(np.mean(recalls)) if recalls else 0.0


def danger_sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Sensitivity/specificity of flagging malignant + precancerous lesions."""
    danger = np.array([c.category in DANGER_CATEGORIES for c in CLASSES])
    t, p = danger[y_true], danger[y_pred]
    sens = float(np.mean(p[t])) if t.any() else 0.0
    spec = float(np.mean(~p[~t])) if (~t).any() else 0.0
    return {"sensitivity": sens, "specificity": spec}


def summarize(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    all_labels = list(range(len(CLASS_KEYS)))
    ham_labels = [CLASS_KEYS.index(k) for k in HAM_CLASSES]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, zero_division=0
    )
    return {
        "n": int(len(y_true)),
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, all_labels),
        "ham_balanced_accuracy": balanced_accuracy(y_true, y_pred, ham_labels),
        "danger": danger_sensitivity(y_true, y_pred),
        "per_class": {
            k: {"precision": float(precision[i]), "recall": float(recall[i]),
                "f1": float(f1[i]), "support": int(support[i])}
            for i, k in enumerate(CLASS_KEYS)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=all_labels).tolist(),
    }
