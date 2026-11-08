"""Post-hoc logit adjustment (Menon et al., 2021) tuned on the validation split.

The network over-predicts frequent classes (67% of HAM10000 is 'nv'). Subtracting
tau * log(prior) from the logits rebalances decisions toward rare classes. A single
scalar tau is chosen on the *validation* split only (never the test split) and the
resulting bias is folded into the classifier layer so the exported model includes it.

Usage:  python -m skinlesion.calibrate
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from skinlesion.config import CHECKPOINT_DIR, CLASS_TO_INDEX, MANIFEST_PATH, NUM_CLASSES
from skinlesion.data import LesionDataset, build_transforms, load_split
from skinlesion.metrics import summarize
from skinlesion.model import load_checkpoint, pick_device
from skinlesion.train import BEST_CHECKPOINT, NUM_WORKERS

log = logging.getLogger(__name__)

TAU_GRID = np.round(np.arange(0.0, 1.51, 0.05), 2)
ROBUST_RADIUS = 2
ADJUSTED_CHECKPOINT = CHECKPOINT_DIR / "best_adjusted.pt"


def log_prior(train_labels: pd.Series) -> np.ndarray:
    counts = np.array([(train_labels == k).sum() for k in CLASS_TO_INDEX], dtype=np.float64)
    if (counts == 0).any():
        raise ValueError("every class needs at least one training example")
    return np.log(counts / counts.sum())


def adjustment(tau: float, prior_log: np.ndarray) -> np.ndarray:
    return -tau * prior_log


def robust_choice(taus: list[float], scores: list[float], radius: int = ROBUST_RADIUS) -> float:
    """Pick the tau whose *worst* score within ±radius grid steps is highest.

    Guards against picking a value sitting next to a cliff where accuracy collapses.
    """
    if not taus or len(taus) != len(scores):
        raise ValueError("taus and scores must be non-empty and equal length")
    worst = [min(scores[max(0, i - radius): i + radius + 1]) for i in range(len(scores))]
    return taus[int(np.argmax(worst))]


def selection_score(summary: dict) -> float:
    return 0.5 * (summary["ham_balanced_accuracy"] + summary["balanced_accuracy"])


@torch.no_grad()
def tta_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    out, targets = [], []
    for images, labels in loader:
        images = images.to(device)
        # Average log-softmax of image + mirror (monotone with the averaged-probability TTA).
        a = torch.log_softmax(model(images), 1)
        b = torch.log_softmax(model(torch.flip(images, dims=[3])), 1)
        out.append(((a + b) / 2).cpu().numpy())
        targets.append(labels.numpy())
    return np.concatenate(out), np.concatenate(targets)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = pick_device()
    manifest = pd.read_csv(MANIFEST_PATH)
    prior_log = log_prior(load_split(manifest, "train").label)
    model = load_checkpoint(str(BEST_CHECKPOINT), device)
    loader = DataLoader(LesionDataset(load_split(manifest, "val"), build_transforms(train=False)),
                        batch_size=64, num_workers=NUM_WORKERS)
    logits, targets = tta_logits(model, loader, device)

    results = []
    for tau in TAU_GRID:
        s = summarize(targets, (logits + adjustment(tau, prior_log)).argmax(1))
        results.append((selection_score(s), float(tau), s))
        log.info("tau=%.2f acc=%.4f bal=%.4f ham_bal=%.4f danger_sens=%.3f spec=%.3f", tau, s["accuracy"],
                 s["balanced_accuracy"], s["ham_balanced_accuracy"], s["danger"]["sensitivity"], s["danger"]["specificity"])
    best_tau = robust_choice([r[1] for r in results], [r[0] for r in results])
    log.info("chosen tau = %.2f", best_tau)

    state = torch.load(BEST_CHECKPOINT, map_location="cpu", weights_only=True)
    bias_key = "head.fc.bias"
    bias = state["model"][bias_key]
    if bias.shape != (NUM_CLASSES,):
        raise RuntimeError(f"unexpected classifier bias shape {tuple(bias.shape)}")
    state["model"][bias_key] = bias + torch.as_tensor(adjustment(best_tau, prior_log), dtype=bias.dtype)
    state["logit_adjustment_tau"] = best_tau
    torch.save(state, ADJUSTED_CHECKPOINT)
    (CHECKPOINT_DIR / "calibration.json").write_text(json.dumps({"tau": best_tau, "log_prior": prior_log.tolist()}, indent=2))
    log.info("wrote %s", ADJUSTED_CHECKPOINT)


if __name__ == "__main__":
    main()
