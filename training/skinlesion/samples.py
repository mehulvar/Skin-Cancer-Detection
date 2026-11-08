"""Pick held-out test images as website samples and copy them into the frontend.

Requires training/reports/deployed_predictions.csv (written by `python -m skinlesion.export`),
so samples are guaranteed to be classified correctly by the exact model the website ships.
Samples are drawn at random (fixed seed) from test images the model classifies
correctly, so the demo reflects behaviour on unseen data; the website's
Performance section reports the full, unfiltered test metrics.

Usage:  python -m skinlesion.samples
"""

from __future__ import annotations

import json
import logging
import shutil

import pandas as pd

from skinlesion.config import CACHE_DIR, PROJECT_ROOT, SEED
from skinlesion.export import DEPLOYED_PREDICTIONS

log = logging.getLogger(__name__)

SAMPLES_DIR = PROJECT_ROOT / "frontend" / "public" / "samples"
SAMPLES_JSON = PROJECT_ROOT / "frontend" / "src" / "data" / "samples.json"

PER_CLASS = {"akiec": 3, "bcc": 3, "bkl": 3, "df": 3, "mel": 4, "nv": 3, "vasc": 3, "other_skin": 5}
NOT_SKIN_PER_SOURCE = {"imagenette": 4, "dtd": 2}
SOURCE_NAMES = {"ham10000": "ISIC 2018 test set", "scin": "SCIN dataset", "imagenette": "Imagenette", "dtd": "Textures (DTD)"}


def choose(preds: pd.DataFrame) -> pd.DataFrame:
    correct = preds[preds.label == preds.pred]
    picks = [
        correct[correct.label == label].sample(n, random_state=SEED)
        for label, n in PER_CLASS.items()
    ]
    not_skin = correct[correct.label == "not_skin"]
    picks += [not_skin[not_skin.source == src].sample(n, random_state=SEED) for src, n in NOT_SKIN_PER_SOURCE.items()]
    return pd.concat(picks, ignore_index=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    preds = pd.read_csv(DEPLOYED_PREDICTIONS)
    chosen = choose(preds)

    if SAMPLES_DIR.exists():
        shutil.rmtree(SAMPLES_DIR)
    SAMPLES_DIR.mkdir(parents=True)
    entries = []
    for i, row in enumerate(chosen.itertuples()):
        name = f"{row.label}-{i:02d}.jpg"
        shutil.copyfile(CACHE_DIR / row.path, SAMPLES_DIR / name)
        entries.append({"id": f"{row.label}-{i:02d}", "src": f"/samples/{name}", "label": row.label,
                        "source": SOURCE_NAMES[row.source]})
    SAMPLES_JSON.write_text(json.dumps(entries, indent=2) + "\n")
    log.info("wrote %d samples to %s", len(entries), SAMPLES_DIR)


if __name__ == "__main__":
    main()
