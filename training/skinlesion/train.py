"""Fine-tune an ImageNet-pretrained EfficientNet on the manifest.

Usage:  python -m skinlesion.train [--epochs 25] [--batch-size 32] [--limit 0]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from skinlesion.config import CHECKPOINT_DIR, MANIFEST_PATH, SEED
from skinlesion.data import LesionDataset, build_transforms, load_split, make_sampler
from skinlesion.metrics import summarize
from skinlesion.model import ARCHITECTURE, build_model, pick_device

log = logging.getLogger(__name__)

LABEL_SMOOTHING = 0.1
WEIGHT_DECAY = 0.02
WARMUP_EPOCHS = 1
NUM_WORKERS = 8
BEST_CHECKPOINT = CHECKPOINT_DIR / "best.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--limit", type=int, default=0, help="debug: cap rows per split")
    return p.parse_args()


def cosine_with_warmup(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, targets = [], []
    for images, labels in loader:
        logits = model(images.to(device, non_blocking=True))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        targets.append(labels.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device) -> float:
    model.train()
    total, count = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        loss = criterion(model(images), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total += loss.item() * len(labels)
        count += len(labels)
    return total / count


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.manual_seed(SEED)
    device = pick_device()

    manifest = pd.read_csv(MANIFEST_PATH)
    train_df, val_df = load_split(manifest, "train"), load_split(manifest, "val")
    if args.limit:
        train_df, val_df = train_df.sample(args.limit, random_state=SEED), val_df.sample(min(args.limit, len(val_df)), random_state=SEED)
    train_ds = LesionDataset(train_df, build_transforms(train=True))
    val_ds = LesionDataset(val_df, build_transforms(train=False))
    loader_kw = dict(num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=make_sampler(train_ds.targets), drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, **loader_kw)
    log.info("device=%s train=%d val=%d", device, len(train_ds), len(val_ds))

    model = build_model(pretrained=True).to(device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: cosine_with_warmup(s, total_steps, WARMUP_EPOCHS * steps_per_epoch)
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_score, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        probs, targets = predict(model, val_loader, device)
        summary = summarize(targets, probs.argmax(1))
        # Select on the mean of HAM balanced accuracy (the hard part) and overall balanced accuracy.
        score = 0.5 * (summary["ham_balanced_accuracy"] + summary["balanced_accuracy"])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": summary["accuracy"],
                        "val_bal_acc": summary["balanced_accuracy"], "val_ham_bal_acc": summary["ham_balanced_accuracy"]})
        log.info("epoch %d/%d loss=%.4f val_acc=%.4f bal_acc=%.4f ham_bal_acc=%.4f (%.0fs)", epoch, args.epochs,
                 train_loss, summary["accuracy"], summary["balanced_accuracy"], summary["ham_balanced_accuracy"], time.time() - start)
        if score > best_score:
            best_score = score
            torch.save({"model": model.state_dict(), "architecture": ARCHITECTURE, "epoch": epoch, "val": summary}, BEST_CHECKPOINT)
            log.info("  saved new best (score=%.4f)", score)
    (CHECKPOINT_DIR / "history.json").write_text(json.dumps(history, indent=2))
    log.info("done; best score %.4f", best_score)


if __name__ == "__main__":
    main()
