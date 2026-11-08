"""Model construction and device selection."""

from __future__ import annotations

import timm
import torch
from torch import nn

from skinlesion.config import NUM_CLASSES

ARCHITECTURE = "convnext_nano.in12k_ft_in1k"
DROP_RATE = 0.2
DROP_PATH_RATE = 0.1


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


class ProbabilityEnsemble(nn.Module):
    """Average member softmax probabilities; output their log so softmax(output) == mean probability.

    Returning log-probabilities keeps the exported graph a drop-in replacement for a single
    model: the website applies softmax to whatever the network outputs.
    """

    def __init__(self, members: list[nn.Module]) -> None:
        super().__init__()
        if not members:
            raise ValueError("an ensemble needs at least one member")
        self.members = nn.ModuleList(members)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = torch.stack([torch.softmax(m(x), dim=1) for m in self.members]).mean(0)
        return torch.log(probs.clamp_min(1e-12))


def _model_from_state(state_dict: dict, architecture: str) -> nn.Module:
    model = build_model(pretrained=False, architecture=architecture)
    model.load_state_dict(state_dict)
    return model


def load_checkpoint(path: str, device: torch.device | str = "cpu") -> nn.Module:
    state = torch.load(path, map_location=device, weights_only=True)
    model = _model_from_state(state["model"], state.get("architecture", ARCHITECTURE))
    return model.to(device).eval()


def load_deployable(path: str, device: torch.device | str = "cpu") -> nn.Module:
    """Load either a single-model checkpoint ('model') or an ensemble checkpoint ('members')."""
    state = torch.load(path, map_location=device, weights_only=True)
    architecture = state.get("architecture", ARCHITECTURE)
    if "members" in state:
        model: nn.Module = ProbabilityEnsemble([_model_from_state(s, architecture) for s in state["members"]])
    elif "model" in state:
        model = _model_from_state(state["model"], architecture)
    else:
        raise ValueError(f"{path} has neither 'model' nor 'members' weights")
    return model.to(device).eval()
