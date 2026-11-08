"""Re-train the ORIGINAL CNN (training/legacy/CNN.py) for an apples-to-apples comparison.

Same architecture (3× Conv-ReLU-MaxPool → Dense(128) → Dropout(0.5) → softmax),
same 128×128 input, Adam, 10 epochs, no class balancing — but trained on the new
lesion-grouped HAM10000 split and scored on the official ISIC 2018 test set.

Usage:  python -m skinlesion.baseline
Writes frontend/src/data/baseline.json.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

from skinlesion.config import HAM_CLASSES, MANIFEST_PATH, PROJECT_ROOT, SEED
from skinlesion.data import LesionDataset, load_split
from skinlesion.metrics import balanced_accuracy
from skinlesion.model import pick_device

log = logging.getLogger(__name__)

LEGACY_SIZE = 128
LEGACY_EPOCHS = 10
LEGACY_BATCH = 32
LEGACY_REPORTED_VAL_ACC = 0.6946  # final epoch in training/legacy/plotResults.py
OUT_PATH = PROJECT_ROOT / "frontend" / "src" / "data" / "baseline.json"


def legacy_cnn(num_classes: int = len(HAM_CLASSES)) -> nn.Module:
    flat = 128 * 14 * 14  # 128 → conv3 126 → pool 63 → conv3 61 → pool 30 → conv3 28 → pool 14
    return nn.Sequential(
        nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(flat, 128), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(128, num_classes),
    )


def per_image_standardization(x: torch.Tensor) -> torch.Tensor:
    """Equivalent of tf.image.per_image_standardization used by the original CNN."""
    return (x - x.mean()) / x.std().clamp(min=1.0 / np.sqrt(x.numel()))


def legacy_transform() -> T.Compose:
    # Original: rescale 1/255 then per-image standardisation.
    return T.Compose([
        T.Resize((LEGACY_SIZE, LEGACY_SIZE)), T.ToTensor(), T.Lambda(per_image_standardization),
    ])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.manual_seed(SEED)
    device = pick_device()
    manifest = pd.read_csv(MANIFEST_PATH)
    ham = manifest[manifest.source == "ham10000"]
    train_ds = LesionDataset(load_split(ham, "train"), legacy_transform())
    test_ds = LesionDataset(load_split(ham, "test"), legacy_transform())
    train_loader = DataLoader(train_ds, batch_size=LEGACY_BATCH, shuffle=True, num_workers=6, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=128, num_workers=6)

    model = legacy_cnn().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, LEGACY_EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        log.info("legacy epoch %d/%d loss=%.4f", epoch, LEGACY_EPOCHS, loss.item())

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds.append(model(x.to(device)).argmax(1).cpu().numpy())
            targets.append(y.numpy())
    y_pred, y_true = np.concatenate(preds), np.concatenate(targets)
    result = {
        "hamAccuracy": float(np.mean(y_pred == y_true)),
        "hamBalancedAccuracy": balanced_accuracy(y_true, y_pred, list(range(len(HAM_CLASSES)))),
        "reportedValAccuracy": LEGACY_REPORTED_VAL_ACC,
    }
    OUT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    log.info("baseline on ISIC 2018 test: %s", result)


if __name__ == "__main__":
    main()
