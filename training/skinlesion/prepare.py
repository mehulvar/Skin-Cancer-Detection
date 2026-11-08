"""Build data/manifest.csv (path,label,source,group,split) with resized cached images.

Splits are leakage-free:
  * HAM10000 train/val are split by lesion_id (the same mole photographed twice
    never appears on both sides — the original CNN.py split did leak this way).
  * The official ISIC 2018 Task 3 test set (1512 images) is the HAM test split;
    the ISIC 2018 validation set (193 images) is added to val.
  * SCIN is split by case_id, OOD images by file.

Usage:  python -m skinlesion.prepare
"""

from __future__ import annotations

import logging
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from skinlesion.config import (
    CACHE_DIR, CACHE_SHORT_SIDE, HAM_CLASSES, MANIFEST_PATH, MAX_OOD_IMAGES, OOD_DIR,
    RAW_DIR, SCIN_DIR, SEED, TEST_FRACTION_EXTRA, VAL_FRACTION,
)
from skinlesion.fetch_scin import resize_short_side

log = logging.getLogger(__name__)

MANIFEST_COLUMNS = ["path", "label", "source", "group", "split"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
RESIZE_WORKERS = 8


def one_hot_to_label(gt: pd.DataFrame) -> pd.DataFrame:
    """Convert an ISIC ground-truth one-hot CSV to (image, label) with lowercase keys."""
    cols = [c.upper() for c in HAM_CLASSES]
    labels = gt[cols].to_numpy().argmax(axis=1)
    return pd.DataFrame({"image": gt["image"], "label": [HAM_CLASSES[i] for i in labels]})


def grouped_split(groups: pd.Series, labels: pd.Series, val_fraction: float, seed: int) -> pd.Series:
    """Assign 'train'/'val' so that no group straddles both, stratified by label."""
    n_splits = max(2, round(1 / val_fraction))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, val_idx = next(splitter.split(np.zeros(len(labels)), labels, groups))
    split = np.full(len(labels), "train", dtype=object)
    split[val_idx] = "val"
    return pd.Series(split, index=labels.index)


def three_way_group_split(groups: pd.Series, val_frac: float, test_frac: float, seed: int) -> pd.Series:
    unique = sorted(groups.unique())
    random.Random(seed).shuffle(unique)
    n_val, n_test = round(len(unique) * val_frac), round(len(unique) * test_frac)
    assignment = {g: "val" for g in unique[:n_val]}
    assignment.update({g: "test" for g in unique[n_val:n_val + n_test]})
    return groups.map(lambda g: assignment.get(g, "train"))


def _cache_image(job: tuple[str, str]) -> bool:
    src, dst = job
    dst_path = Path(dst)
    if dst_path.exists():
        return True
    try:
        with Image.open(src) as img:
            resize_short_side(img.convert("RGB"), CACHE_SHORT_SIDE).save(dst_path, "JPEG", quality=92)
        return True
    except (OSError, ValueError) as exc:
        log.warning("skipping unreadable %s: %s", src, exc)
        return False


def cache_images(sources: list[Path], subdir: str) -> list[Path | None]:
    out_dir = CACHE_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [out_dir / f"{i:06d}_{s.stem}.jpg" for i, s in enumerate(sources)]
    with ProcessPoolExecutor(max_workers=RESIZE_WORKERS) as pool:
        ok = list(pool.map(_cache_image, [(str(s), str(t)) for s, t in zip(sources, targets)], chunksize=64))
    return [t if good else None for t, good in zip(targets, ok)]


def build_ham() -> pd.DataFrame:
    meta = pd.read_csv(RAW_DIR / "HAM10000_metadata.tab", sep="\t")
    train_gt = one_hot_to_label(pd.read_csv(RAW_DIR / "ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv"))
    train = train_gt.merge(meta[["image_id", "lesion_id"]], left_on="image", right_on="image_id", how="left")
    if train["lesion_id"].isna().any():
        raise ValueError("HAM10000 metadata is missing lesion_id for some training images")
    train = train.assign(
        src=[RAW_DIR / "ISIC2018_Task3_Training_Input" / f"{i}.jpg" for i in train.image],
        group=train.lesion_id,
        split=grouped_split(train.lesion_id, train.label, VAL_FRACTION, SEED),
    )

    def official(name: str, split: str) -> pd.DataFrame:
        gt = one_hot_to_label(pd.read_csv(RAW_DIR / f"ISIC2018_Task3_{name}_GroundTruth/ISIC2018_Task3_{name}_GroundTruth.csv"))
        return gt.assign(
            src=[RAW_DIR / f"ISIC2018_Task3_{name}_Input" / f"{i}.jpg" for i in gt.image],
            group=gt.image, split=split,
        )

    frame = pd.concat([train, official("Validation", "val"), official("Test", "test")], ignore_index=True)
    return frame.assign(source="ham10000")[["src", "label", "source", "group", "split"]]


def build_scin() -> pd.DataFrame:
    sel = pd.read_csv(SCIN_DIR / "selected.csv").drop_duplicates(subset="file")
    sel = sel[sel.file.map(lambda f: Path(f).exists())]
    return pd.DataFrame({
        "src": sel.file.map(Path), "label": "other_skin", "source": "scin",
        "group": sel.case_id.astype(str),
        "split": three_way_group_split(sel.case_id.astype(str), VAL_FRACTION, TEST_FRACTION_EXTRA, SEED),
    })


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def build_ood() -> pd.DataFrame:
    imagenette = list_images(OOD_DIR / "imagenette2-320")
    dtd = list_images(OOD_DIR / "dtd" / "images")
    rng = random.Random(SEED)
    # Half everyday objects/scenes, half textures (hard negatives: blotchy, spotted, ...).
    half = MAX_OOD_IMAGES // 2
    picked = rng.sample(imagenette, min(half, len(imagenette))) + rng.sample(dtd, min(half, len(dtd)))
    groups = pd.Series([str(p) for p in picked])
    return pd.DataFrame({
        "src": picked, "label": "not_skin",
        "source": ["imagenette" if "imagenette" in str(p) else "dtd" for p in picked],
        "group": groups, "split": three_way_group_split(groups, VAL_FRACTION, TEST_FRACTION_EXTRA, SEED),
    })


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parts = []
    for name, builder in (("ham", build_ham), ("scin", build_scin), ("ood", build_ood)):
        frame = builder()
        cached = cache_images(list(frame.src), name)
        frame = frame.assign(path=[str(c.relative_to(CACHE_DIR)) if c else None for c in cached]).dropna(subset=["path"])
        parts.append(frame[MANIFEST_COLUMNS])
        log.info("%s: %d images", name, len(frame))
    manifest = pd.concat(parts, ignore_index=True)
    manifest.to_csv(MANIFEST_PATH, index=False)
    log.info("wrote %s\n%s", MANIFEST_PATH, manifest.groupby(["split", "label"]).size().unstack(0).to_string())


if __name__ == "__main__":
    main()
