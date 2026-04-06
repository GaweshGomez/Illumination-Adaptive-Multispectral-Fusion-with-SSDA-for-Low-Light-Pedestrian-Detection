# Adaptive Multispectral Fusion for Low-Light Pedestrian Detection with Semi-Supervised Domain Adaptation

A two-part deep learning system that trains a pedestrian detector on paired RGB-thermal nighttime imagery and adapts it to a new thermal-only target domain using only 18.1% of available labelled images — achieving a **1,433% improvement** over zero-shot transfer.

---

## Overview

This is a final year research project for BSc (Hons) Artificial Intelligence & Data Science at Robert Gordon University, Aberdeen. The system addresses a practical deployment constraint: **training uses paired RGB + Thermal images (LLVIP), but inference uses thermal only (CVC-14 Night)** — exactly how most real-world thermal cameras are deployed.

The pipeline combines two major components:

| Component | Technology | Purpose |
|---|---|---|
| Part A — Source Domain Fusion | YOLOv8n + Illumination-Adaptive Sigmoid | Train a pedestrian detector on illumination-adaptively blended RGB+Thermal LLVIP images |
| Part B — SSDA Pipeline | YOLOv8n + IoC Pseudo-Label Filter | Adapt the source model to CVC-14 Night thermal domain using 614 GT labels and 1,972 unlabelled images |

**Starting from mAP@0.5 = 0.0005** (near-total domain collapse, zero-shot), the final model achieves **mAP@0.5 = 0.7164** on the held-out CVC-14 NewTest set using only 614 labelled target images (18.1% of the full training pool), running entirely on the Kaggle free-tier Tesla T4 GPU.

---

## Repository Structure

```
adaptive-multispectral-fusion-ssda/
│
├── notebooks/
│   ├── 01_adaptive_fusion_llvip.ipynb       ← Part A: Adaptive fusion training on LLVIP
│   ├── 02_simplefusionv2.ipynb              ← Part A: Corrected Simple Fusion 50/50 baseline
│   ├── 03_rgb_thermal_baseline.ipynb        ← Part A: RGB-Only and Thermal-Only baselines
│   └── 04_cvc14_ssda_pipeline.ipynb         ← Part B: Full SSDA pipeline (Cells 1–51)  ← START HERE
│
├── docs/
│   ├── Chapter_4_Dataset_Preparation.md     ← LLVIP, CVC-14, preprocessing, domain gap
│   ├── Chapter_5_System_Design.md           ← Fusion module, SSDA pipeline, IoC filter
│   └── assets/
│       └── system_architecture.png          ← Full system architecture diagram
│
├── results/
│   └── final_results_VERIFIED.csv           ← All 11 experiments, verified metrics
│
├── models/
│   └── README.md                            ← Model download instructions (weights on Kaggle)
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Key Results

### Part A — Fusion Ablation (LLVIP Validation Set)

| Method | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Selected |
|---|---|---|---|---|---|
| RGB-Only Baseline | 0.8946 | 0.5255 | 0.9210 | 0.8245 | |
| Simple Fusion (50/50) | 0.7815 | 0.4472 | 0.8518 | 0.7769 | — Below RGB baseline |
| Thermal-Only Baseline | 0.9625 | 0.6600 | 0.9597 | 0.9016 | |
| Adaptive Fusion V1 | 0.9596 | 0.6174 | 0.9521 | 0.9000 | |
| **Adaptive Fusion V2 (FINAL)** | **0.9634** | **0.6449** | **0.9626** | **0.9001** | Yes — Source Model |

> **Key finding:** Simple Fusion (0.7815) falls **below both** single-modality baselines. Naive 50/50 blending at night dilutes thermal contrast, actively hurting performance. Adaptive Fusion V2 outperforms Simple Fusion by **+0.1819 mAP@0.5** — the primary empirical argument for illumination-adaptive weighting.

### Part B — SSDA Pipeline (CVC-14 NewTest — 727 images, 1,894 GT instances)

| Stage | GT Labels | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Notes |
|---|---|---|---|---|---|---|
| Zero-Shot Transfer | 0 | 0.0005 | 0.0001 | 0.000 | 0.000 | Near-total domain collapse |
| Pure SSL (no GT) | 0 | 0.0010 | — | — | — | Failed — noise amplification |
| Warm-Start | 114 | 0.3918* | — | 0.4160* | 0.4386* | *train-val metric (biased) |
| SSDA Round 1 | 114 | 0.5202 | 0.2250 | 0.4924 | 0.5222 | Naive pseudo-labels |
| Phase 2 — IoC SSDA | 614 | 0.6264 | 0.3066 | 0.6485 | 0.5873 | IoC filter applied |
| Phase 3A — Best Recall | 614 | 0.6770 | 0.3279 | 0.6851 | 0.6869 | Highest recall |
| **Phase 3B — FINAL** | **614** | **0.7164** | **0.3655** | **0.7149** | **0.6452** | Thermal augmentations |

> *Warm-start metric is train-val (biased by temporal adjacency in video data). All other mAP@0.5 values are on the held-out CVC-14 NewTest partition (different recording session, no temporal overlap).*

### Two Recommended Models

| Model | mAP@0.5 | Precision | Recall | Best For |
|---|---|---|---|---|
| `phase3thermalbest.pt` | **0.7164** | 0.7149 | 0.6452 | Best overall accuracy |
| `phase3iocr2best.pt` | 0.6770 | 0.6851 | **0.6869** | Safety-critical deployment (best recall) |

---

## Prerequisites

Before you begin, make sure you have:

- A free [Kaggle](https://www.kaggle.com/) account
- GPU enabled on Kaggle — **Settings → Accelerator → GPU T4 x1**
- Internet enabled on Kaggle — **Settings → Internet → On**
- Basic familiarity with running Jupyter notebook cells

> All pipeline notebooks are designed to run on Kaggle due to GPU and dataset storage requirements. Local execution requires a CUDA-capable GPU. See [Local Setup](#local-setup-optional) below.

---

## Quick Start — Run the Full SSDA Pipeline (Recommended)

Follow these 5 steps to reproduce the final Part B results on CVC-14.

### Step 1 — Clone or Download This Repository

```bash
git clone https://github.com/gaweshgomes/adaptive-multispectral-fusion-ssda.git
```

Or click **Code → Download ZIP** on GitHub and extract it.

### Step 2 — Attach the Required Kaggle Datasets

Go to your Kaggle notebook settings and attach these two datasets:

| Dataset | Kaggle Path | Contents |
|---|---|---|
| [adaptive-fusion-v2-model](https://www.kaggle.com/datasets/gaweshgomes/adaptive-fusion-v2-model) | `kaggle/input/adaptive-fusion-v2-model/` | `best.pt` — LLVIP Adaptive Fusion V2 source model |
| [ssda-cvc14-complete-results-models](https://www.kaggle.com/datasets/gaweshgomes/ssda-cvc14-complete-results-models) | `kaggle/input/ssda-cvc14-complete-results-models/` | All CVC-14 SSDA checkpoints — Phase 2, 3A, 3B, LPLD |
| [rgb-thermal-baseline-results-models](https://www.kaggle.com/datasets/gaweshgomes/rgb-thermal-baseline-results-models) | `kaggle/input/rgb-thermal-baseline-results-models/` | RGB-Only and Thermal-Only baseline models |
| [simple-fusion-v2-results-models](https://www.kaggle.com/datasets/gaweshgomes/simple-fusion-v2-results-models) | `kaggle/input/simple-fusion-v2-results-models/` | Simple Fusion 50/50 corrected model (epoch 54) |

### Step 3 — Upload the Pipeline Notebook to Kaggle

1. Go to [Kaggle](https://www.kaggle.com/) → **Create → New Notebook**
2. Click the three-dot menu **(⋮) → Import Notebook**
3. Upload `notebooks/04_cvc14_ssda_pipeline.ipynb` from this repo
4. Confirm settings in the right panel:
   - **Accelerator** → GPU T4 x1
   - **Internet** → On
   - Confirm all three datasets above are attached

### Step 4 — Apply the PyTorch Compatibility Patch

> **Critical:** PyTorch 2.6+ changed `torch.load()` default to `weights_only=True`, breaking all Ultralytics `.pt` loading. This patch must be in every cell that loads a model.

```python
import ultralytics.nn.tasks as tasks

def patched_torch_safe_load(weight):
    from ultralytics.utils.downloads import attempt_download_asset
    file = attempt_download_asset(weight)
    ckpt = torch.load(file, map_location='cpu', weights_only=False)
    return ckpt, file

tasks.torch_safe_load = patched_torch_safe_load
```

This patch is already included in all notebook cells that load `.pt` files.

### Step 5 — Run the Notebook Cells in Order

Click **Run All** or run each cell manually in this sequence:

| Cell | What it does | Expected Time |
|---|---|---|
| Cell 1 | Installs dependencies, applies PyTorch patch | ~30 s |
| Cell 2 | Verifies GPU, prints device info | ~5 s |
| Cell 3 | Converts CVC-14 annotations (640×471 resolution), generates fused images | ~3 min |
| Cell 8–10 | Loads LLVIP source model, runs zero-shot evaluation on NewTest | ~2 min |
| Cell 11 | Centre-offset diagnostic (domain gap analysis) | ~1 min |
| Cell 15–20 | Warm-start fine-tuning on 114 GT labels + 802 hard negatives | ~10 min |
| Cell 25–30 | SSDA Round 1 naive pseudo-labelling | ~12 min |
| Cell 38–39 | Pool audit + IoC flip-consistency filter | ~5 min |
| Cell 40 | Phase 2 IoC-SSDA training (614 GT + 256 IoC pseudo-labels) | ~8 min |
| Cell 43 | Phase 3A — second SSDA round, no thermal aug | ~8 min |
| Cell 48 | Phase 3B — thermal-specific augmentations (FINAL MODEL) | ~6 min |
| Cell 50–51 | LPLD experiment (quality ceiling finding) | ~5 min |

> **Total runtime:** approximately 60–75 minutes on T4 GPU for a full end-to-end run.

---

## Running Other Notebooks

### Part A — Adaptive Fusion Training on LLVIP (Reproduce Source Model)

Requires the [LLVIP dataset](https://www.kaggle.com/datasets/gaweshgomes/llvip-rgb-thermal-yolo-format) (~12,025 paired training images, ~6.5 GB).

| Notebook | Kaggle Link | What it does |
|---|---|---|
| `01_adaptive_fusion_llvip.ipynb` | [adaptive-fusion-v02-1-2](https://www.kaggle.com/code/gaweshgomes/adaptive-fusion-v02-1-2) | Per-image illumination-adaptive fusion, YOLOv8n training, 100 epochs |
| `02_simplefusionv2.ipynb` | [simplefusionv2](https://www.kaggle.com/code/gaweshgomes/simplefusionv2) | Corrected Simple Fusion 50/50 baseline (epoch 54 best) |
| `03_rgb_thermal_baseline.ipynb` | [rgb-thermal-simplefusion](https://www.kaggle.com/code/gaweshgomes/rgb-thermal-simplefusion) | RGB-Only and Thermal-Only baselines |

Saved results and model weights for each Part A experiment:

| Experiment | Results + Model Dataset |
|---|---|
| RGB-Only Baseline & Thermal-Only Baseline | [rgb-thermal-baseline-results-models](https://www.kaggle.com/datasets/gaweshgomes/rgb-thermal-baseline-results-models) |
| Simple Fusion V2 (50/50 corrected) | [simple-fusion-v2-results-models](https://www.kaggle.com/datasets/gaweshgomes/simple-fusion-v2-results-models) |
| Adaptive Fusion V2 (FINAL source model) | [adaptive-fusion-v2-model](https://www.kaggle.com/datasets/gaweshgomes/adaptive-fusion-v2-model) |

Steps:
1. Open the relevant notebook link on Kaggle
2. Attach the [LLVIP dataset](https://www.kaggle.com/datasets/gaweshgomes/llvip-rgb-thermal-yolo-format) as a data source
3. Set **Accelerator → GPU T4** and **Internet → On**
4. Click **Run All** — best checkpoint saves to `/kaggle/working/`

> Training time: approximately 3–4 hours per notebook on T4 GPU.

### Part B — Zero-Shot and Domain Analysis (Reproduce Baseline Diagnostics)

All experiments including zero-shot evaluation, histogram matching, and centre-offset analysis are in **Cells 1–15** of `04_cvc14_ssda_pipeline.ipynb`. These cells run in under 10 minutes.

---

## Model Weights

Model weights are not stored in this repository due to file size. All checkpoints are hosted on Kaggle.

| Model | File | mAP@0.5 | Size | Download |
|---|---|---|---|---|
| LLVIP Source Model (Adaptive Fusion V2) | `best.pt` | 0.9634 (LLVIP val) | ~5.95 MB | [adaptive-fusion-v2-model](https://www.kaggle.com/datasets/gaweshgomes/adaptive-fusion-v2-model) |
| RGB-Only Baseline | `rgb_baseline_best.pt` | 0.8946 (LLVIP val) | ~5.95 MB | [rgb-thermal-baseline-results-models](https://www.kaggle.com/datasets/gaweshgomes/rgb-thermal-baseline-results-models) |
| Thermal-Only Baseline | `thermal_baseline_best.pt` | 0.9625 (LLVIP val) | ~5.95 MB | [rgb-thermal-baseline-results-models](https://www.kaggle.com/datasets/gaweshgomes/rgb-thermal-baseline-results-models) |
| Simple Fusion V2 (50/50) | `simplefusion_best.pt` | 0.7815 (LLVIP val) | ~5.95 MB | [simple-fusion-v2-results-models](https://www.kaggle.com/datasets/gaweshgomes/simple-fusion-v2-results-models) |
| Phase 3B FINAL — Best mAP | `phase3thermalbest.pt` | 0.7164 (CVC-14 NewTest) | ~5.95 MB | [ssda-cvc14-complete-results-models](https://www.kaggle.com/datasets/gaweshgomes/ssda-cvc14-complete-results-models) |
| Phase 3A — Best Recall | `phase3iocr2best.pt` | 0.6770 / R=0.6869 | ~5.95 MB | [ssda-cvc14-complete-results-models](https://www.kaggle.com/datasets/gaweshgomes/ssda-cvc14-complete-results-models) |
| Phase 2 — IoC SSDA | `phase2best.pt` | 0.6264 (CVC-14 NewTest) | ~5.95 MB | [ssda-cvc14-complete-results-models](https://www.kaggle.com/datasets/gaweshgomes/ssda-cvc14-complete-results-models) |
| SSDA Round 1 | `ssdaround1best.pt` | 0.5202 (CVC-14 NewTest) | ~5.95 MB | [ssda-cvc14-complete-results-models](https://www.kaggle.com/datasets/gaweshgomes/ssda-cvc14-complete-results-models) |

After downloading, place `.pt` files in the `models/` folder when running locally.

---

## Local Setup (Optional)

Requires a CUDA-capable GPU. CPU-only mode is not recommended for training.

```bash
# 1. Clone the repository
git clone https://github.com/gaweshgomes/adaptive-multispectral-fusion-ssda.git
cd adaptive-multispectral-fusion-ssda

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download model weights from Kaggle and place in models/

# 5. Launch the pipeline notebook
jupyter notebook notebooks/04_cvc14_ssda_pipeline.ipynb
```

> **Note:** The datasets (LLVIP ~13 GB, CVC-14 ~2 GB) must be downloaded from Kaggle and placed at the paths expected by the notebooks. See `docs/Chapter_4_Dataset_Preparation.md` for full dataset structure.

---

## System Architecture

```
Input: LLVIP Paired RGB + Thermal Images (12,025 training pairs)
        │
        ▼
┌──────────────────────────────────────────┐
│  PART A — SOURCE DOMAIN (LLVIP)          │
│                                          │
│  Per-Image Illumination-Adaptive Fusion  │
│  rgb_w = 1 / (1 + exp(-(illum*4 - 1)))  │
│  fused = rgb_w*RGB + (1-rgb_w)*Thermal   │
│                                          │
│  YOLOv8n Training                        │
│  100 epochs · SGD · 640×640 · Batch 16  │
│  mAP@0.5 = 0.9634 on LLVIP val          │
└─────────────────┬────────────────────────┘
                  │  best.pt transferred
                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  PART B — TARGET DOMAIN SSDA (CVC-14 Night)                              │
│                                                                          │
│  Stage 0 │ Zero-Shot Eval          │ 0 GT   │ mAP@0.5 = 0.0005          │
│     ↓                                                                    │
│  Stage 1 │ Pure SSL                │ 0 GT   │ mAP@0.5 = 0.0010 [FAILED] │
│     ↓                                                                    │
│  Stage 2 │ Warm-Start Fine-Tune   │ 114 GT  │ mAP@0.5 = 0.3918*         │
│     ↓                                                                    │
│  Stage 3 │ SSDA Round 1 Naive PL  │ 114 GT  │ mAP@0.5 = 0.5202          │
│     ↓                                                                    │
│  Stage 4 │ IoC Consistency Filter  │        │ 390 HC boxes / 2,847 raw   │
│          │ Flip-consistency check  │        │ 13% coverage, clean labels │
│     ↓                                                                    │
│  Stage 5 │ Phase 2 IoC-SSDA       │ 614 GT  │ mAP@0.5 = 0.6264          │
│     ↓                                                                    │
│  Stage 6 │ Phase 3A (Best Recall) │ 614 GT  │ mAP@0.5 = 0.6770          │
│     ↓                                                                    │
│  Stage 7 │ Phase 3B Thermal Augs  │ 614 GT  │ mAP@0.5 = 0.7164 [FINAL]  │
└──────────────────────────────────────────────────────────────────────────┘

Evaluation: CVC-14 NewTest — 727 images · 1,894 GT instances
            Temporally disjoint from training (different recording session)
            Primary metric: mAP@0.5  |  Safety metric: Recall
```

---

## Author

**Gawesh Gomes**
BSc (Hons) Artificial Intelligence & Data Science
Robert Gordon University, Aberdeen, UK | Informatics Institute of Technology, Sri Lanka
RUG ID: 2313082 | IIT ID: 20220578
Kaggle: [@gaweshgomes](https://www.kaggle.com/gaweshgomes)
