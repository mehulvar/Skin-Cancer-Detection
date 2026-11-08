"""Evaluate the best checkpoint on the held-out test split and write reports.

Usage:  python -m skinlesion.evaluate
Outputs training/reports/test_metrics.json and confusion_matrix.png.
"""

from __future__ import annotations

import json
import logging

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from skinlesion.config import CLASS_KEYS, MANIFEST_PATH, REPORTS_DIR
from skinlesion.data import LesionDataset, build_transforms, load_split
from skinlesion.metrics import summarize
from skinlesion.model import load_checkpoint, pick_device
from skinlesion.train import BEST_CHECKPOINT, NUM_WORKERS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

log = logging.getLogger(__name__)
EVAL_BATCH_SIZE = 64


@torch.no_grad()
def predict_with_flip_tta(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (plain_probs, tta_probs, targets) — TTA averages the image and its h-flip."""
    plain, tta, targets = [], [], []
    for images, labels in loader:
        images = images.to(device)
        p = torch.softmax(model(images), 1)
        pf = torch.softmax(model(torch.flip(images, dims=[3])), 1)
        plain.append(p.cpu().numpy())
        tta.append(((p + pf) / 2).cpu().numpy())
        targets.append(labels.numpy())
    return np.concatenate(plain), np.concatenate(tta), np.concatenate(targets)


def plot_confusion(cm: list[list[int]], path) -> None:
    matrix = np.asarray(cm, dtype=float)
    norm = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)
    plt.figure(figsize=(10, 8))
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASS_KEYS, yticklabels=CLASS_KEYS)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Test confusion matrix (row-normalised)")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = pick_device()
    model = load_checkpoint(str(BEST_CHECKPOINT), device)
    test_df = load_split(pd.read_csv(MANIFEST_PATH), "test")
    loader = DataLoader(LesionDataset(test_df, build_transforms(train=False)), batch_size=EVAL_BATCH_SIZE, num_workers=NUM_WORKERS)
    plain, tta, targets = predict_with_flip_tta(model, loader, device)

    report = {"plain": summarize(targets, plain.argmax(1)), "flip_tta": summarize(targets, tta.argmax(1))}
    ham_mask = test_df.source.to_numpy() == "ham10000"
    report["ham_test_only"] = summarize(targets[ham_mask], tta[ham_mask].argmax(1))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "test_metrics.json").write_text(json.dumps(report, indent=2))
    plot_confusion(report["flip_tta"]["confusion_matrix"], REPORTS_DIR / "confusion_matrix.png")
    for name in ("plain", "flip_tta"):
        r = report[name]
        log.info("%s: acc=%.4f bal_acc=%.4f ham_bal_acc=%.4f danger_sens=%.4f danger_spec=%.4f", name,
                 r["accuracy"], r["balanced_accuracy"], r["ham_balanced_accuracy"],
                 r["danger"]["sensitivity"], r["danger"]["specificity"])
    for k, v in report["flip_tta"]["per_class"].items():
        log.info("  %-11s recall=%.3f precision=%.3f f1=%.3f n=%d", k, v["recall"], v["precision"], v["f1"], v["support"])


if __name__ == "__main__":
    main()
