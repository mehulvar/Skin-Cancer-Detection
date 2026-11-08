"""Unit tests for the pure parts of the training pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from skinlesion.config import CLASS_KEYS, CLASS_TO_INDEX, CLASSES, IMAGE_SIZE, NUM_CLASSES
from skinlesion.data import center_square_resize, load_split, sample_weights
from skinlesion.fetch_scin import is_non_neoplastic, parse_condition_list, resize_short_side
from skinlesion.metrics import balanced_accuracy, danger_sensitivity, summarize
from skinlesion.prepare import grouped_split, one_hot_to_label, three_way_group_split
from skinlesion.train import cosine_with_warmup


def test_class_table_is_consistent():
    assert NUM_CLASSES == 9
    assert CLASS_KEYS[:7] == ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
    assert all(CLASS_TO_INDEX[c.key] == i for i, c in enumerate(CLASSES))


@pytest.mark.parametrize("conditions, expected", [
    (["Eczema", "Psoriasis"], True),
    (["Keratosis pilaris"], True),
    (["Eczema", "Melanoma"], False),
    (["SK/ISK"], False),
    (["Actinic Keratosis", "Eczema"], False),
    (["Basal Cell Carcinoma"], False),
    ([], False),
])
def test_scin_filter_excludes_lesion_lookalikes(conditions, expected):
    assert is_non_neoplastic(conditions) is expected


def test_parse_condition_list_handles_bad_input():
    assert parse_condition_list("['Eczema', 'Tinea']") == ["Eczema", "Tinea"]
    assert parse_condition_list("[]") == []
    assert parse_condition_list(float("nan")) == []
    assert parse_condition_list("not a list(") == []


def test_one_hot_to_label_maps_isic_columns():
    gt = pd.DataFrame({"image": ["a", "b"], "MEL": [1.0, 0.0], "NV": [0.0, 0.0], "BCC": [0.0, 0.0],
                       "AKIEC": [0.0, 0.0], "BKL": [0.0, 0.0], "DF": [0.0, 0.0], "VASC": [0.0, 1.0]})
    assert one_hot_to_label(gt).label.tolist() == ["mel", "vasc"]


def test_grouped_split_never_leaks_a_lesion_across_splits():
    rng = np.random.default_rng(0)
    groups = pd.Series([f"L{i // 2}" for i in range(400)])  # two images per lesion
    labels = pd.Series(rng.choice(["nv", "mel", "bkl"], size=400))
    split = grouped_split(groups, labels, val_fraction=0.15, seed=0)
    per_group = pd.DataFrame({"g": groups, "s": split}).groupby("g").s.nunique()
    assert (per_group == 1).all()
    assert 0.08 < (split == "val").mean() < 0.25


def test_three_way_split_is_grouped_and_deterministic():
    groups = pd.Series([f"C{i // 3}" for i in range(300)])
    a = three_way_group_split(groups, 0.15, 0.15, seed=1)
    b = three_way_group_split(groups, 0.15, 0.15, seed=1)
    assert a.equals(b)
    assert set(a) == {"train", "val", "test"}
    assert (pd.DataFrame({"g": groups, "s": a}).groupby("g").s.nunique() == 1).all()


def test_center_square_resize_output_size_and_center():
    img = Image.new("RGB", (400, 200), (255, 0, 0))
    img.paste((0, 255, 0), (100, 0, 300, 200))  # green centre square
    out = center_square_resize(img)
    assert out.size == (IMAGE_SIZE, IMAGE_SIZE)
    assert out.getpixel((IMAGE_SIZE // 2, IMAGE_SIZE // 2)) == (0, 255, 0)


def test_resize_short_side_never_upscales():
    small = Image.new("RGB", (100, 80))
    assert resize_short_side(small, 300).size == (100, 80)
    assert resize_short_side(Image.new("RGB", (900, 600)), 300).size == (450, 300)


def test_sample_weights_balance_rare_classes():
    w = sample_weights([0, 0, 0, 0, 1], power=1.0)
    assert w[:4].sum() == pytest.approx(w[4])


def test_balanced_accuracy_is_mean_recall():
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([0, 0, 1, 1])
    assert balanced_accuracy(y_true, y_pred, [0, 1]) == pytest.approx((2 / 3 + 1) / 2)


def test_danger_sensitivity_counts_malignant_and_precancerous():
    mel, nv, akiec = CLASS_TO_INDEX["mel"], CLASS_TO_INDEX["nv"], CLASS_TO_INDEX["akiec"]
    y_true = np.array([mel, akiec, nv, nv])
    y_pred = np.array([akiec, nv, nv, mel])
    d = danger_sensitivity(y_true, y_pred)
    assert d["sensitivity"] == pytest.approx(0.5)
    assert d["specificity"] == pytest.approx(0.5)


def test_summarize_has_full_confusion_matrix():
    y = np.arange(NUM_CLASSES)
    s = summarize(y, y)
    assert s["accuracy"] == 1.0
    assert np.array(s["confusion_matrix"]).shape == (NUM_CLASSES, NUM_CLASSES)


def test_load_split_rejects_unknown_labels():
    frame = pd.DataFrame({"path": ["x"], "label": ["bogus"], "split": ["train"]})
    with pytest.raises(ValueError, match="unknown labels"):
        load_split(frame, "train")
    with pytest.raises(ValueError, match="no rows"):
        load_split(frame, "val")


def test_cosine_with_warmup_shape():
    assert cosine_with_warmup(0, 100, 10) == pytest.approx(0.1)
    assert cosine_with_warmup(10, 100, 10) == pytest.approx(1.0)
    assert cosine_with_warmup(100, 100, 10) == pytest.approx(0.0)


def test_model_forward_shape():
    from skinlesion.model import build_model
    model = build_model(pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert out.shape == (1, NUM_CLASSES)
