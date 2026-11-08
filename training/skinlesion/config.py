"""Shared configuration: class definitions, paths and training constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
SCIN_DIR = DATA_DIR / "scin"
OOD_DIR = DATA_DIR / "ood"
MANIFEST_PATH = DATA_DIR / "manifest.csv"

TRAINING_DIR = PROJECT_ROOT / "training"
CHECKPOINT_DIR = TRAINING_DIR / "checkpoints"
REPORTS_DIR = TRAINING_DIR / "reports"
FRONTEND_MODEL_DIR = PROJECT_ROOT / "frontend" / "public" / "model"


@dataclass(frozen=True)
class LesionClass:
    key: str
    name: str
    category: str  # "malignant" | "precancerous" | "benign" | "non-lesion" | "invalid"


# Index order is the model's output order. Do not reorder without retraining.
CLASSES: tuple[LesionClass, ...] = (
    LesionClass("akiec", "Actinic keratosis / intraepithelial carcinoma", "precancerous"),
    LesionClass("bcc", "Basal cell carcinoma", "malignant"),
    LesionClass("bkl", "Benign keratosis-like lesion", "benign"),
    LesionClass("df", "Dermatofibroma", "benign"),
    LesionClass("mel", "Melanoma", "malignant"),
    LesionClass("nv", "Melanocytic nevus (mole)", "benign"),
    LesionClass("vasc", "Vascular lesion", "benign"),
    LesionClass("other_skin", "Other skin condition (non-cancerous)", "non-lesion"),
    LesionClass("not_skin", "Not a skin image", "invalid"),
)
CLASS_KEYS: tuple[str, ...] = tuple(c.key for c in CLASSES)
CLASS_TO_INDEX: dict[str, int] = {k: i for i, k in enumerate(CLASS_KEYS)}
HAM_CLASSES: tuple[str, ...] = CLASS_KEYS[:7]
NUM_CLASSES = len(CLASSES)

# Preprocessing — must match frontend/src/lib/preprocess.ts exactly.
IMAGE_SIZE = 256
CACHE_SHORT_SIDE = 300
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Split fractions
VAL_FRACTION = 0.15
TEST_FRACTION_EXTRA = 0.15  # for SCIN / OOD sources (HAM uses official ISIC test set)
SEED = 42

# Caps for auxiliary classes so they don't dominate training.
MAX_SCIN_IMAGES = 4500
MAX_OOD_IMAGES = 6000
