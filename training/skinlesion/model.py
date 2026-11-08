"""Model construction and device selection."""

from __future__ import annotations

import timm
import torch
from torch import nn

from skinlesion.config import NUM_CLASSES

ARCHITECTURE = "efficientnet_b2"
DROP_RATE = 0.3
DROP_PATH_RATE = 0.2


def build_model(pretrained: bool = True, architecture: str = ARCHITECTURE) -> nn.Module:
    return timm.create_model(
        architecture, pretrained=pretrained, num_classes=NUM_CLASSES,
        drop_rate=DROP_RATE, drop_path_rate=DROP_PATH_RATE,
    )


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path: str, device: torch.device | str = "cpu") -> nn.Module:
    state = torch.load(path, map_location=device, weights_only=True)
    model = build_model(pretrained=False, architecture=state.get("architecture", ARCHITECTURE))
    model.load_state_dict(state["model"])
    return model.to(device).eval()
