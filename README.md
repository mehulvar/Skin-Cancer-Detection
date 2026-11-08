# SkinSight — Skin Lesion Classification

**Live demo:** https://skinsight-ruddy.vercel.app

SkinSight classifies dermoscopic images of skin lesions into the seven HAM10000
categories, and also recognises **non-cancerous skin conditions** and **images
that aren't skin at all**. The model runs entirely in the browser (ONNX Runtime
Web), so uploaded images never leave the user's device.

> ⚠️ Educational research project — not a medical device and not a diagnosis.

## Results

All numbers are measured on held-out data. HAM10000 lesions are scored on the
**official ISIC 2018 Task 3 test set** (1,512 images from lesions never seen in
training). The website shows exactly these metrics, computed from the deployed
(exported ONNX) model.

| ISIC 2018 test set (7 lesion classes) | Original CNN | SkinSight |
|---|---|---|
| Accuracy | 66.3% | 81.4% |
| Balanced accuracy (official ISIC metric) | 33.3% | 75.3% |

| Full 9-class test set (3,080 images) | |
|---|---|
| Accuracy | 90.5% |
| Balanced accuracy | 80.6% |
| Malignant/precancerous sensitivity | 82.7% |
| Malignant/precancerous specificity | 95.0% |

Per-class recall on the test set:

| Class | Recall |
|---|---|
| Actinic keratosis / Bowen's (precancerous) | 51.2% |
| Basal cell carcinoma | 79.6% |
| Benign keratosis | 76.0% |
| Dermatofibroma | 81.8% |
| Melanoma | 75.4% |
| Melanocytic nevus (mole) | 85.6% |
| Vascular lesion | 77.1% |
| Other skin condition (SCIN) | 99.4% |
| Not skin | 99.1% |

The "Original CNN" column re-trains the exact architecture from
`training/legacy/CNN.py` (3 conv layers, 128 px, Adam, 10 epochs) on the same
lesion-grouped split, for a fair comparison. Its originally reported ~70 %
validation accuracy came from a random split in which rotated copies of the
same lesion could appear in both training and validation.

## Classes

| Key | Class | Category |
|---|---|---|
| `akiec` | Actinic keratosis / intraepithelial carcinoma | Precancerous |
| `bcc` | Basal cell carcinoma | Malignant |
| `bkl` | Benign keratosis-like lesion | Benign |
| `df` | Dermatofibroma | Benign |
| `mel` | Melanoma | Malignant |
| `nv` | Melanocytic nevus (mole) | Benign |
| `vasc` | Vascular lesion | Benign |
| `other_skin` | Other skin condition — eczema, dermatitis, urticaria, psoriasis, acne… | Non-cancerous |
| `not_skin` | Not a skin image — everyday objects, scenes, textures | Invalid input |

## What changed from the original project

| | Original | Now |
|---|---|---|
| Model | 3-layer CNN from scratch, 128 px | ImageNet-12k pretrained ConvNeXt-Nano, fine-tuned at 224 px |
| Splits | Random 80/20; rotated duplicates leaked across splits | Grouped by `lesion_id`; official ISIC 2018 test set held out |
| Imbalance | None (67 % of images are moles) | Square-root class-balanced sampling + label smoothing |
| Augmentation | Offline rotations of 3 classes | Online random crops, flips, rotations, colour jitter, blur, erasing |
| Out-of-scope inputs | Always predicts a lesion | `other_skin` + `not_skin` classes |
| Deployment | — | Next.js site on Vercel, in-browser WASM inference with flip TTA |

## Data

| Source | Used for | Images |
|---|---|---|
| [HAM10000](https://doi.org/10.7910/DVN/DBW86T) / [ISIC 2018 Task 3](https://challenge.isic-archive.com/data/#2018) | 7 lesion classes (train/val + official test) | 10,015 + 193 + 1,512 |
| [SCIN](https://github.com/google-research-datasets/scin) (Google) | `other_skin` — only cases whose entire dermatologist differential is non-neoplastic | 4,493 |
| [Imagenette](https://github.com/fastai/imagenette) + [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/) | `not_skin` — objects, scenes, textures | 6,000 |

The original Google Drive dataset is not required — everything is downloaded
from the public sources above.

## Repository layout

```
training/
  skinlesion/         Python package
    config.py         class table, paths, constants
    fetch_scin.py     download filtered SCIN images
    prepare.py        build data/manifest.csv with leakage-free splits + resized cache
    data.py           dataset, transforms, balanced sampler
    model.py          ConvNeXt-Nano via timm
    train.py          fine-tuning loop (AMP on Apple MPS / CUDA)
    evaluate.py       test metrics, confusion matrix, per-image predictions
    export.py         ONNX export, int8 quantisation gate, publish to frontend
    samples.py        pick held-out test images as website samples
    baseline.py       re-train the original CNN for comparison
  tests/              pytest unit tests
  legacy/             the original scripts (CNN.py, GAN, plots, …)
  reports/            test_metrics.json, confusion_matrix.png
frontend/             Next.js 16 app (deployed to Vercel)
  src/lib/            preprocessing, inference, verdict logic (+ vitest tests)
  src/components/     page sections
  e2e/                Playwright end-to-end tests
  public/model/       skin_lesion.onnx, confusion_matrix.png
  public/samples/     sample images from the held-out test sets
```

## Reproducing

```bash
cd training
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt

# 1. Data (≈3.5 GB) — raw download commands below
python -m skinlesion.fetch_scin          # SCIN subset
python -m skinlesion.prepare             # manifest + cache

# 2. Train, evaluate, export
python -m skinlesion.train --epochs 25   # ~1 h on an Apple M-series GPU
python -m skinlesion.evaluate
python -m skinlesion.baseline
python -m skinlesion.export
python -m skinlesion.samples

pytest                                   # unit tests
```

Raw data downloads (run inside `data/raw`; Imagenette goes in `data/ood`):

```bash
B=https://isic-archive.s3.amazonaws.com/challenges/2018
for f in Training_Input Training_GroundTruth Validation_Input Validation_GroundTruth Test_Input Test_GroundTruth; do
  curl -LO $B/ISIC2018_Task3_$f.zip && unzip -q ISIC2018_Task3_$f.zip; done
curl -L -o HAM10000_metadata.tab https://dataverse.harvard.edu/api/access/datafile/4338392
curl -LO https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
# DTD: https://huggingface.co/datasets/tanganke/dtd (parquet) → data/ood/dtd/images/*.jpg
# SCIN metadata: https://storage.googleapis.com/dx-scin-public-data/dataset/scin_{cases,labels}.csv → data/scin
```

### Frontend

```bash
cd frontend
npm install
npm run dev          # http://localhost:3000
npm test             # unit tests (vitest)
npm run test:e2e     # Playwright against a production build
E2E_BASE_URL=https://skinsight-ruddy.vercel.app npm run test:e2e   # same suite against the live site
```

The Vercel project uses `frontend/` as its root directory and deploys on every
push to `main`.

## Limitations

- HAM10000 is mostly fair-skinned patients from Austria and Australia; accuracy
  on darker skin tones is not established.
- The lesion classes were trained on **dermoscopic** images. Ordinary phone
  photos of moles look different and are less reliable.
- `other_skin` and `not_skin` cover common cases but cannot cover everything a
  user might upload.
- Not clinically validated. Always see a doctor about a worrying spot.

## Credits

Data: Tschandl et al., *The HAM10000 dataset* (2018); ISIC Archive; Google
Research SCIN; fast.ai Imagenette; Cimpoi et al., DTD. Used for non-commercial
research and education under their respective licences.
