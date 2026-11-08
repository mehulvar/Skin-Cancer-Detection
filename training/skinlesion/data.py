"""Dataset, transforms and sampling."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms as T

from skinlesion.config import CACHE_DIR, CLASS_TO_INDEX, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

# Sampling weight ∝ count^-SAMPLING_POWER (0 = natural distribution, 1 = fully balanced).
SAMPLING_POWER = 0.75


def center_square_resize(img: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """Center-crop to a square on the short side, then bilinear-resize.

    Mirrors frontend/src/lib/preprocess.ts so browser and training inputs match.
    """
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    square = img.crop((left, top, left + side, top + side))
    return square.resize((size, size), Image.Resampling.BILINEAR)


def build_transforms(train: bool) -> T.Compose:
    normalize = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        return T.Compose([T.Lambda(center_square_resize), *normalize])
    return T.Compose([
        T.RandomResizedCrop(IMAGE_SIZE, scale=(0.4, 1.0), ratio=(0.75, 1.333)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomApply([T.RandomRotation(180)], p=0.5),
        T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.03),
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.15),
        *normalize,
        T.RandomErasing(p=0.2, scale=(0.02, 0.12)),
    ])


def load_split(manifest: pd.DataFrame, split: str) -> pd.DataFrame:
    frame = manifest[manifest.split == split].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"manifest has no rows for split '{split}'")
    unknown = set(frame.label) - set(CLASS_TO_INDEX)
    if unknown:
        raise ValueError(f"unknown labels in manifest: {sorted(unknown)}")
    return frame


class LesionDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: T.Compose, root: Path = CACHE_DIR) -> None:
        self.paths = [str(root / p) for p in frame.path]
        self.targets = [CLASS_TO_INDEX[label] for label in frame.label]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[idx]) as img:
            tensor = self.transform(img.convert("RGB"))
        return tensor, self.targets[idx]


def sample_weights(targets: list[int], power: float = SAMPLING_POWER) -> np.ndarray:
    counts = Counter(targets)
    return np.array([counts[t] ** -power for t in targets], dtype=np.float64)


def make_sampler(targets: list[int]) -> WeightedRandomSampler:
    weights = torch.as_tensor(sample_weights(targets))
    return WeightedRandomSampler(weights, num_samples=len(targets), replacement=True)
