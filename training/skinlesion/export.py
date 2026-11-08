"""Export the best checkpoint to ONNX, optionally quantise, and publish to the frontend.

Steps:
  1. Export FP32 ONNX and verify parity with PyTorch.
  2. Dynamically quantise weights to int8 (≈4× smaller download).
  3. Score both ONNX models on the full test split with flip TTA (exactly what the
     website runs) and ship the quantised one only if it loses ≤ QUANT_MAX_DROP
     balanced accuracy.
  4. Write frontend/public/model/skin_lesion.onnx, confusion_matrix.png and
     frontend/src/data/model-meta.json with the *deployed* model's test metrics.

Usage:  python -m skinlesion.export
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch.utils.data import DataLoader

from skinlesion.config import (
    CHECKPOINT_DIR, CLASS_KEYS, FRONTEND_MODEL_DIR, IMAGE_SIZE, MANIFEST_PATH, PROJECT_ROOT, REPORTS_DIR,
)
from skinlesion.data import LesionDataset, build_transforms, load_split
from skinlesion.evaluate import plot_confusion
from skinlesion.metrics import summarize
from skinlesion.model import ARCHITECTURE, load_checkpoint
from skinlesion.calibrate import ADJUSTED_CHECKPOINT
from skinlesion.train import BEST_CHECKPOINT

log = logging.getLogger(__name__)
ONNX_NAME = "skin_lesion.onnx"
OPSET = 17
PARITY_TOLERANCE = 1e-3
QUANT_MAX_DROP = 0.01
EVAL_BATCH = 32
DEPLOYED_PREDICTIONS = REPORTS_DIR / "deployed_predictions.csv"
META_PATH = PROJECT_ROOT / "frontend" / "src" / "data" / "model-meta.json"
ARCH_DISPLAY = {"convnext_nano.in12k_ft_in1k": "ConvNeXt-Nano (ImageNet-12k pretrained)"}


def deployed_checkpoint() -> Path:
    """The logit-adjusted checkpoint if calibration ran, else the raw best checkpoint."""
    return ADJUSTED_CHECKPOINT if ADJUSTED_CHECKPOINT.exists() else BEST_CHECKPOINT


def export_fp32(path: Path, checkpoint: Path) -> None:
    model = load_checkpoint(str(checkpoint), "cpu")
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(
        model, dummy, str(path), input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=OPSET, dynamo=False,
    )
    sample = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        expected = model(sample).numpy()
    actual = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"input": sample.numpy()})[0]
    max_diff = float(np.abs(expected - actual).max())
    if max_diff > PARITY_TOLERANCE:
        raise RuntimeError(f"ONNX output differs from PyTorch by {max_diff}")
    log.info("fp32 export ok (%.1f MB), max |diff| = %.2e", path.stat().st_size / 1e6, max_diff)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def predict_onnx(path: Path, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Flip-TTA probabilities from an ONNX model — the same computation the website runs."""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    loader = DataLoader(LesionDataset(test_df, build_transforms(train=False)), batch_size=EVAL_BATCH, num_workers=6)
    probs, targets = [], []
    for images, labels in loader:
        x = images.numpy()
        p = softmax(session.run(None, {"input": x})[0])
        pf = softmax(session.run(None, {"input": np.ascontiguousarray(x[..., ::-1])})[0])
        probs.append((p + pf) / 2)
        targets.append(labels.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def headline(full: dict, ham: dict, model_mb: float) -> dict:
    return {
        "testImages": full["n"], "accuracy": full["accuracy"], "balancedAccuracy": full["balanced_accuracy"],
        "hamBalancedAccuracy": ham["ham_balanced_accuracy"], "hamAccuracy": ham["accuracy"], "hamTestImages": ham["n"],
        "dangerSensitivity": full["danger"]["sensitivity"], "dangerSpecificity": full["danger"]["specificity"],
        "perClassRecall": {k: v["recall"] for k, v in full["per_class"].items()},
        "modelMb": round(model_mb, 1), "architecture": ARCH_DISPLAY.get(ARCHITECTURE, ARCHITECTURE),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    checkpoint = deployed_checkpoint()
    log.info("exporting %s", checkpoint.name)
    fp32_path, int8_path = CHECKPOINT_DIR / "model_fp32.onnx", CHECKPOINT_DIR / "model_int8.onnx"
    export_fp32(fp32_path, checkpoint)
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)

    test_df = load_split(pd.read_csv(MANIFEST_PATH), "test")
    ham_mask = (test_df.source == "ham10000").to_numpy()
    results = {}
    for name, path in (("fp32", fp32_path), ("int8", int8_path)):
        probs, y_true = predict_onnx(path, test_df)
        y_pred = probs.argmax(1)
        full, ham = summarize(y_true, y_pred), summarize(y_true[ham_mask], y_pred[ham_mask])
        results[name] = (path, full, ham, probs)
        log.info("%s: %.1f MB acc=%.4f bal=%.4f ham_acc=%.4f ham_bal=%.4f", name, path.stat().st_size / 1e6,
                 full["accuracy"], full["balanced_accuracy"], ham["accuracy"], ham["ham_balanced_accuracy"])

    drop = results["fp32"][2]["ham_balanced_accuracy"] - results["int8"][2]["ham_balanced_accuracy"]
    chosen = "int8" if drop <= QUANT_MAX_DROP else "fp32"
    path, full, ham, probs = results[chosen]
    log.info("shipping %s (ham balanced-accuracy drop from quantisation: %.4f)", chosen, drop)

    FRONTEND_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, FRONTEND_MODEL_DIR / ONNX_NAME)
    plot_confusion(full["confusion_matrix"], FRONTEND_MODEL_DIR / "confusion_matrix.png")
    meta = {"imageSize": IMAGE_SIZE, "quantization": chosen, "metrics": headline(full, ham, path.stat().st_size / 1e6)}
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "deployed_metrics.json").write_text(json.dumps({"full": full, "ham": ham, "variant": chosen}, indent=2))
    test_df.assign(pred=[CLASS_KEYS[i] for i in probs.argmax(1)], confidence=probs.max(1)).to_csv(
        DEPLOYED_PREDICTIONS, index=False)


if __name__ == "__main__":
    main()
