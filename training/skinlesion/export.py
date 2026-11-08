"""Export the best checkpoint to ONNX for in-browser inference and verify parity.

Usage:  python -m skinlesion.export
Writes frontend/public/model/{skin_lesion.onnx, labels.json}.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import onnxruntime as ort
import torch

from skinlesion.config import CLASSES, FRONTEND_MODEL_DIR, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, REPORTS_DIR
from skinlesion.model import load_checkpoint
from skinlesion.train import BEST_CHECKPOINT

log = logging.getLogger(__name__)
ONNX_NAME = "skin_lesion.onnx"
OPSET = 17
PARITY_TOLERANCE = 1e-3


def export(out_dir=FRONTEND_MODEL_DIR) -> None:
    model = load_checkpoint(str(BEST_CHECKPOINT), "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / ONNX_NAME
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(
        model, dummy, str(onnx_path), input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=OPSET, dynamo=False,
    )

    sample = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        expected = model(sample).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"input": sample.numpy()})[0]
    max_diff = float(np.abs(expected - actual).max())
    if max_diff > PARITY_TOLERANCE:
        raise RuntimeError(f"ONNX output differs from PyTorch by {max_diff}")
    log.info("exported %s (%.1f MB), max |diff| = %.2e", onnx_path, onnx_path.stat().st_size / 1e6, max_diff)

    metrics_path = REPORTS_DIR / "test_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    labels = {
        "imageSize": IMAGE_SIZE, "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
        "classes": [{"key": c.key, "name": c.name, "category": c.category} for c in CLASSES],
        "metrics": _headline(metrics) if metrics else None,
    }
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2))


def _headline(metrics: dict) -> dict:
    tta, ham = metrics["flip_tta"], metrics["ham_test_only"]
    return {
        "testImages": tta["n"], "accuracy": tta["accuracy"], "balancedAccuracy": tta["balanced_accuracy"],
        "hamBalancedAccuracy": ham["ham_balanced_accuracy"], "hamAccuracy": ham["accuracy"],
        "dangerSensitivity": tta["danger"]["sensitivity"], "dangerSpecificity": tta["danger"]["specificity"],
        "perClassRecall": {k: v["recall"] for k, v in tta["per_class"].items()},
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    export()
