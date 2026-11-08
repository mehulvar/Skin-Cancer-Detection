"""Download a filtered subset of Google's SCIN dataset as the 'other_skin' class.

Only cases whose entire dermatologist differential is non-neoplastic are kept
(eczema, dermatitis, urticaria, psoriasis, acne, ...), so nothing that could be
a skin cancer or one of the HAM10000 lesion types ends up labelled 'other_skin'.

Usage:  python -m skinlesion.fetch_scin
"""

from __future__ import annotations

import ast
import io
import logging
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

from skinlesion.config import CACHE_SHORT_SIDE, MAX_SCIN_IMAGES, SCIN_DIR, SEED

log = logging.getLogger(__name__)

SCIN_BUCKET_URL = "https://storage.googleapis.com/dx-scin-public-data/"
DOWNLOAD_WORKERS = 16
DOWNLOAD_TIMEOUT_S = 60

# Any differential containing one of these is excluded (lesion / neoplasm look-alikes).
EXCLUDE_KEYWORDS: tuple[str, ...] = (
    "melan", "nev", "naev", "carcinoma", "scc", "bcc", "actinic", "sk/isk",
    "seborrheic keratosis", "lentig", "dermatofibroma", "hemangioma", "angioma",
    "kaposi", "lymphoma", "tumor", "neoplasm", "cyst", "mole", "keratoacanthoma",
    "pyogenic granuloma", "skin tag", "acrochordon", "lipoma", "malignan",
    "bowen", "mycosis fungoides", "porokeratosis",
)


def is_non_neoplastic(conditions: list[str]) -> bool:
    """True when a non-empty differential contains no lesion/neoplasm look-alikes."""
    if not conditions:
        return False
    lowered = [c.lower() for c in conditions]
    return not any(k in c for c in lowered for k in EXCLUDE_KEYWORDS)


def parse_condition_list(raw: object) -> list[str]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    return [str(x) for x in parsed] if isinstance(parsed, list) else []


def select_cases(cases: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (case_id, image_path) for eligible cases."""
    merged = cases.merge(labels, on="case_id")
    gradable = merged["dermatologist_gradable_for_skin_condition_1"].astype(str).str.startswith("DEFAULT_YES")
    conditions = merged["dermatologist_skin_condition_on_label_name"].map(parse_condition_list)
    eligible = merged[gradable & conditions.map(is_non_neoplastic)]
    rows = [
        {"case_id": str(r.case_id), "image_path": getattr(r, col)}
        for r in eligible.itertuples()
        for col in ("image_1_path", "image_2_path", "image_3_path")
        if isinstance(getattr(r, col), str)
    ]
    return pd.DataFrame(rows, columns=["case_id", "image_path"])


def resize_short_side(img: Image.Image, short_side: int) -> Image.Image:
    w, h = img.size
    scale = short_side / min(w, h)
    if scale >= 1:
        return img.copy()
    return img.resize((round(w * scale), round(h * scale)), Image.Resampling.BICUBIC)


def download_one(image_path: str, out_path: Path) -> bool:
    if out_path.exists():
        return True
    try:
        with urllib.request.urlopen(SCIN_BUCKET_URL + image_path, timeout=DOWNLOAD_TIMEOUT_S) as resp:
            payload = resp.read()
        with Image.open(io.BytesIO(payload)) as img:
            resize_short_side(img.convert("RGB"), CACHE_SHORT_SIDE).save(out_path, "JPEG", quality=92)
        return True
    except Exception as exc:  # network / decode errors: skip the image, keep going
        log.warning("failed %s: %s", image_path, exc)
        return False


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cases = pd.read_csv(SCIN_DIR / "scin_cases.csv")
    labels = pd.read_csv(SCIN_DIR / "scin_labels.csv")
    selected = select_cases(cases, labels)
    if len(selected) > MAX_SCIN_IMAGES:
        selected = selected.sample(n=MAX_SCIN_IMAGES, random_state=SEED)
    log.info("selected %d images from %d cases", len(selected), selected.case_id.nunique())

    image_dir = SCIN_DIR / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    targets = [image_dir / f"{Path(p).stem}.jpg" for p in selected.image_path]
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        ok = list(pool.map(download_one, selected.image_path, targets))

    index = selected.assign(file=[str(t) for t in targets], ok=ok)
    index[index.ok].drop(columns="ok").to_csv(SCIN_DIR / "selected.csv", index=False)
    log.info("downloaded %d / %d", sum(ok), len(ok))


if __name__ == "__main__":
    main()
