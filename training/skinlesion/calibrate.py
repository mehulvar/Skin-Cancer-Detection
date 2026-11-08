"""Post-hoc logit adjustment (Menon et al., 2021) tuned on the validation split.

The network over-predicts frequent classes (67% of HAM10000 is 'nv'). Subtracting
tau * log(prior) from the logits rebalances decisions toward rare classes. A single
scalar tau is chosen on the *validation* split only (never the test split) and the
resulting bias is folded into the classifier layer so the exported model includes it.

All trained runs found (best_run1.pt, best.pt) are combined into a probability-averaging
ensemble; the same tau is folded into every member.

Usage:  python -m skinlesion.calibrate
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from skinlesion.config import CHECKPOINT_DIR, CLASS_TO_INDEX, MANIFEST_PATH, NUM_CLASSES
from skinlesion.data import LesionDataset, build_transforms, load_split
from skinlesion.metrics import summarize
from skinlesion.model import ARCHITECTURE, load_checkpoint, pick_device
from skinlesion.train import BEST_CHECKPOINT, NUM_WORKERS

log = logging.getLogger(__name__)

TAU_GRID = np.round(np.arange(0.0, 1.51, 0.05), 2)
ROBUST_RADIUS = 2
BIAS_KEY = "head.fc.bias"
ADJUSTED_CHECKPOINT = CHECKPOINT_DIR / "best_adjusted.pt"
# An earlier, separate training run of the same recipe. Ensembling it with the latest run raised
# validation selection score from 0.781 to 0.795 (both with flip TTA).
PREVIOUS_RUN_CHECKPOINT = CHECKPOINT_DIR / "best_run1.pt"


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


def fold_adjustment(state_dict: dict, shift: np.ndarray) -> dict:
    """Return a copy of state_dict with `shift` added to the classifier bias."""
    bias = state_dict[BIAS_KEY]
    if bias.shape != (NUM_CLASSES,):
        raise RuntimeError(f"unexpected classifier bias shape {tuple(bias.shape)}")
    return {**state_dict, BIAS_KEY: bias + torch.as_tensor(shift, dtype=bias.dtype)}


def ensemble_members() -> list[Path]:
    """Every trained run that exists (earlier run first), so calibration ensembles all of them."""
    members = [p for p in (PREVIOUS_RUN_CHECKPOINT, BEST_CHECKPOINT) if p.exists()]
    if not members:
        raise FileNotFoundError(f"no trained checkpoint found in {CHECKPOINT_DIR}")
    return members


def ensemble_val_logits(members: list[Path], loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    """Flip-TTA log-probabilities averaged over the ensemble members."""
    per_member = []
    for path in members:
        logits, targets = tta_logits(load_checkpoint(str(path), device), loader, device)
        per_member.append(logits)
        log.info("scored %s on val", path.name)
    return np.mean(per_member, axis=0), targets


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = pick_device()
    manifest = pd.read_csv(MANIFEST_PATH)
    prior_log = log_prior(load_split(manifest, "train").label)
    loader = DataLoader(LesionDataset(load_split(manifest, "val"), build_transforms(train=False)),
                        batch_size=64, num_workers=NUM_WORKERS)
    members = ensemble_members()
    logits, targets = ensemble_val_logits(members, loader, device)

    results = []
    for tau in TAU_GRID:
        s = summarize(targets, (logits + adjustment(tau, prior_log)).argmax(1))
        results.append((selection_score(s), float(tau), s))
        log.info("tau=%.2f acc=%.4f bal=%.4f ham_bal=%.4f danger_sens=%.3f spec=%.3f", tau, s["accuracy"],
                 s["balanced_accuracy"], s["ham_balanced_accuracy"], s["danger"]["sensitivity"], s["danger"]["specificity"])
    best_tau = robust_choice([r[1] for r in results], [r[0] for r in results])
    log.info("chosen tau = %.2f", best_tau)

    shift = adjustment(best_tau, prior_log)
    states = [torch.load(p, map_location="cpu", weights_only=True) for p in members]
    torch.save({
        "members": [fold_adjustment(s["model"], shift) for s in states],
        "architecture": states[-1].get("architecture", ARCHITECTURE),
        "logit_adjustment_tau": best_tau,
        "member_files": [p.name for p in members],
    }, ADJUSTED_CHECKPOINT)
    (CHECKPOINT_DIR / "calibration.json").write_text(json.dumps(
        {"tau": best_tau, "log_prior": prior_log.tolist(), "members": [p.name for p in members]}, indent=2))
    log.info("wrote %s (%d-model ensemble)", ADJUSTED_CHECKPOINT, len(members))


if __name__ == "__main__":
    main()
