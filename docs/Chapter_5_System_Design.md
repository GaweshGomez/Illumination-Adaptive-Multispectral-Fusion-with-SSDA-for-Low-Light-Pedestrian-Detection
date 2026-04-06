# Chapter 5: System Design

> **Module:** CM4605 Individual Research Project  
> **Student:** Gawesh Gomes · RUG ID: 2313082  
> **Project:** Adaptive Multispectral Fusion for Low-Light Pedestrian Detection with Semi-Supervised Domain Adaptation  
> **Platform:** Kaggle Notebooks · Tesla T4 GPU · PyTorch 2.9.0+cu126 · Ultralytics YOLOv8 8.4.23

---

## 5.1 Overall Architecture

The system is divided into two sequential, self-contained components:

```
┌─────────────────────────────────────────────────────────────────────┐
│  PART A · Source Domain (LLVIP)                                     │
│                                                                     │
│  LLVIP Paired RGB+Thermal  →  Illumination-Adaptive Fusion          │
│                            →  YOLOv8n Training (fused images)       │
│                            →  best.pt  (Source Model)               │
└─────────────────────────────────────┬───────────────────────────────┘
                                      │  best.pt transferred
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PART B · Target Domain SSDA (CVC-14 Night)                         │
│                                                                     │
│  Zero-Shot Eval  →  Warm-Start (114 GT)  →  SSDA R1 (Naive PL)      │
│  →  IoC Filter  →  Phase 2 (IoC-SSDA)  →  Phase 3A (Best Recall)   │
│  →  Phase 3B Thermal Augs  →  FINAL MODEL (mAP@0.5 = 0.7164)       │
└─────────────────────────────────────────────────────────────────────┘
```

**Key design constraint:** At inference time on CVC-14, only the **thermal (FIR) channel** is available. The system must generalise from multispectral training (LLVIP: RGB+Thermal) to single-modality inference (CVC-14: Thermal only).

---

## 5.2 Part A: Illumination-Adaptive Fusion Module

### 5.2.1 The Fusion Formula

The core contribution of Part A is a closed-form, per-image illumination-adaptive sigmoid blending rule:

```python
illumination = mean(RGB_float)                    # scalar in [0.0, 1.0]
rgb_w = 1.0 / (1.0 + exp(-(illumination * 4.0 - 1.0)))   # sigmoid gate

fused = rgb_w * RGB_float + (1 - rgb_w) * Thermal_float
fused_uint8 = clip(fused * 255, 0, 255).astype(uint8)
```

| Condition | Illumination | rgb_w | Dominant Modality |
|---|---|---|---|
| Very dark night | 0.17 | 0.44 | Thermal (56%) |
| Moderately dark | 0.50 | 0.73 | RGB (73%) |
| Less dark | 0.70 | 0.87 | RGB (87%) |

> **Key property:** Each image receives its own unique rgb_w value. Simple Fusion (50/50) fixes rgb_w = 0.50 for all images regardless of conditions, which actively hurts performance by diluting thermal contrast.

### 5.2.2 Structural Duality (Important Distinction)

The Part A notebook (`adaptive-fusion-v02-1-2.ipynb`) contains two distinct model architectures:

| Component | Cells | Status | Description |
|---|---|---|---|
| Feature-Level Dual-Stream | Cells 4–6 | **Designed, NOT trained** | Full dual-stream YOLOv8n with Channel + Spatial Attention |
| Pixel-Level Fusion (DEPLOYED) | Cells 8–10 | **Trained and deployed** | Standard YOLOv8n on pixel-fused images |

The Ultralytics `model.train()` API accepts only a single 3-channel image stream. Training the dual-stream model would require a fully custom training loop. This is documented as **future work**.

The **deployed model** is a standard `YOLOv8n` (3,011,043 parameters) fine-tuned on pixel-level illumination-adaptively blended LLVIP images.

### 5.2.3 Designed Feature-Level Architecture (Not Trained)

```
RGB Input  ──►  YOLOv8n Backbone (layers 0–8)  ──►  RGB Features (256ch, 20×20)
                                                           │
                                               ChannelAttention(F_rgb)
                                                           │
Thermal Input ► YOLOv8n Backbone (layers 0–8)  ──►  Thermal Features (256ch, 20×20)
                                                           │
                                               SpatialAttention(F_therm)
                                                           │
                                         IlluminationAdaptiveFusion
                                         rgb_w = sigmoid(illumination * temp + bias)
                                         fused = rgb_w * Attend(F_rgb) + therm_w * Attend(F_therm)
                                                           │
                                              ──►  YOLOv8n FPN Neck + Head
```

- **ChannelAttention** (F_rgb): Squeeze-and-Excitation over 256 feature channels — selects which channels carry useful RGB signal.
- **SpatialAttention** (F_therm): 7×7 conv over avg+max maps — identifies hot pedestrian body regions in the spatial domain.
- **Learnable parameters:** `temp` (init=4.0), `bias` (init=–1.0) — total fusion overhead: 8,564 parameters.
- **Skip connections** (layers 4, 6) come from the **thermal stream only** — thermal is the stronger modality at night.

### 5.2.4 Part A Training Configuration

```yaml
base_model:   yolov8n.pt
data:         data.yaml  (fused LLVIP images, single class: person)
epochs:       100
patience:     20
imgsz:        640
batch:        16
optimizer:    SGD
lr0:          0.01
lrf:          0.01
momentum:     0.937
weight_decay: 0.0005
augmentations:
  hsv_h: 0.005
  hsv_s: 0.3
  hsv_v: 0.2
  translate: 0.1
  scale: 0.5
  fliplr: 0.5
  mosaic: 1.0
```

### 5.2.5 Part A Ablation Results (LLVIP Validation Set)

| Method | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|
| RGB-Only Baseline | 0.8946 | 0.5255 | 0.9210 | 0.8245 |
| Simple Fusion (50/50) | 0.7815 | 0.4472 | 0.8518 | 0.7769 |
| Thermal-Only Baseline | 0.9625 | 0.6600 | 0.9597 | 0.9016 |
| Adaptive Fusion V1 | 0.9596 | 0.6174 | 0.9521 | 0.9000 |
| **Adaptive Fusion V2 (FINAL)** | **0.9634** | **0.6449** | **0.9626** | **0.9001** |

> **Critical finding:** Simple Fusion (0.7815) falls **below both** single-modality baselines. Naïve 50/50 blending dilutes thermal pedestrian-background contrast (2550 → 152⁄15), producing a domain-confused image harder to process than either modality alone. Adaptive Fusion V2 outperforms Simple Fusion by **+0.1819 mAP@0.5**.

---

## 5.3 Part B: Semi-Supervised Domain Adaptation (SSDA) Pipeline

### 5.3.1 Pipeline Stages

```
Stage 0 │ Zero-Shot Transfer         │ 0 GT labels    │ mAP@0.5 = 0.0005
   ↓
Stage 1 │ Pure SSL (no GT)           │ 0 GT labels    │ mAP@0.5 = 0.0010  [FAILED]
   ↓
Stage 2 │ Warm-Start Fine-Tuning     │ 114 GT labels  │ mAP@0.5 = 0.3918* [anchor]
   ↓
Stage 3 │ SSDA Round 1 (Naive PL)    │ 114 GT labels  │ mAP@0.5 = 0.5202
   ↓
Stage 4 │ SSDA Round 2 (SSL Satur.)  │ 114 GT labels  │ mAP@0.5 = 0.4158  [REGRESSED]
   ↓
Stage 5 │ Phase 2 — IoC-SSDA         │ 614 GT labels  │ mAP@0.5 = 0.6264
   ↓
Stage 6 │ Phase 3A — Best Recall     │ 614 GT labels  │ mAP@0.5 = 0.6770
   ↓
Stage 7 │ Phase 3B — Thermal Augs    │ 614 GT labels  │ mAP@0.5 = 0.7164  [FINAL BEST]
```

*Stage 2 Warm-Start metric is train-val biased (temporal adjacency). All other mAP@0.5 values on CVC-14 NewTest (727 images, 1,894 GT instances).*

### 5.3.2 IoC Pseudo-Label Filter (Core V2 Contribution)

The **IoC (Image over Consistency)** filter exploits bilateral symmetry of pedestrian bodies to separate genuine detections from noise.

**Algorithm:**

```
For each unlabelled Pool A image:
  Pass 1: Run model on ORIGINAL image         → boxes B1, confidences C1
  Pass 2: Run model on HORIZONTALLY FLIPPED   → boxes B2_flip
  Un-flip B2_flip to original space           → boxes B2

  For each box b1 in B1:
    best_iou = max(IoU(b1, b2) for b2 in B2)

    if C1 ≥ 0.30  AND  best_iou ≥ 0.50  →  HC label (hard pseudo-label)
    if C1 ≥ 0.10  AND  best_iou ≥ 0.50  →  LC box  (LPLD pool)
    otherwise                            →  REJECTED
```

**IoC Filter Survival Funnel (actual output, Cell 39):**

| Stage | Count |
|---|---|
| Raw detections (conf ≥ 0.25) | 2,847 boxes |
| After conf ≥ 0.30 filter | 1,821 boxes |
| After IoC consistency (IoU ≥ 0.50) | 542 boxes |
| Final HC labels (training) | 390 boxes |
| LC boxes (LPLD pool) | 781 boxes |
| Pool A images with HC labels | 256 / 1,972 (13.0% coverage) |

**Quality vs Quantity Result:**

| Approach | Coverage | Pseudo-boxes | mAP@0.5 |
|---|---|---|---|
| V1 Naïve (conf > 0.50) | 33.1% (421/1,272 images) | 800 boxes | 0.5202 |
| V2 IoC Filtered | 13.0% (256/1,972 images) | 390 boxes | 0.6264 |

> Fewer but cleaner pseudo-labels produce **+20.4 mAP@0.5 absolute gain**. The bottleneck is label quality, not volume.

### 5.3.3 Thermal-Specific Augmentations (Phase 3B)

Standard YOLO augmentations include hue and saturation shifts, which are meaningless for grayscale thermal images. Phase 3B replaces them with thermal-physically-motivated augmentations:

| Augmentation | V1/V2 Value | Phase 3B Value | Rationale |
|---|---|---|---|
| `hsv_h` (hue shift) | 0.015 | **0.0** | Thermal has no colour/hue |
| `hsv_s` (saturation) | 0.7 | **0.0** | Thermal is grayscale |
| `hsv_v` (brightness) | 0.4 | **0.35** | Keep — thermal intensity varies with temperature |
| `fliplr` | 0.5 | 0.5 | Keep — pedestrians appear same under horizontal flip |
| `erasing` | 0.4 | **0.2** | Reduce — pedestrians already sparse |
| `degrees` | 0.0 | 0.0 | No rotation — CVC-14 is fixed-mount bus camera |
| CLAHE | — | **applied** | Simulates thermal contrast variation |

### 5.3.4 Training Configurations by Phase

| Phase | Base Model | Epochs | lr0 | Patience | Optimiser |
|---|---|---|---|---|---|
| Warm-Start | LLVIP best.pt | 60 | 0.0005 | 15 | AdamW |
| SSDA Round 1 | Warm-start best.pt | 60 | 0.0005 | 15 | AdamW |
| Phase 2 IoC | SSDA R1 best.pt | 40 | 0.0005 | 10 | AdamW |
| Phase 3A | Phase 2 best.pt | 40 | 0.0003 | 12 | AdamW |
| Phase 3B | Phase 2 best.pt | 40 (ES@25) | 0.0003 | 12 | AdamW |

### 5.3.5 Final Model Performance (CVC-14 NewTest)

Two models are recommended for reporting:

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Best For |
|---|---|---|---|---|---|
| Phase 3B — Thermal Augs | **0.7164** | 0.3655 | 0.7149 | 0.6452 | Best overall mAP |
| Phase 3A — No Thermal Aug | 0.6770 | 0.3279 | 0.6851 | **0.6869** | Safety-critical (best recall) |

> **Headline result:** Starting from mAP@0.5 = 0.0005 (zero-shot), the final model achieves **0.7164 — a 1,433% improvement** using only 614 GT labels (18.1% of the full training pool), on a free Kaggle Tesla T4 GPU.

---

## 5.4 LPLD Experiment (Negative Result — Research Contribution)

**Low-Confidence Pseudo-Label Distillation (LPLD)** was applied to mine the 781 LC boxes from the IoC pool using a soft cross-entropy loss.

| Metric | Before LPLD | After LPLD |
|---|---|---|
| mAP@0.5 | 0.7164 | 0.6189 |
| Recall | 0.6452 | 0.5892 |

**Finding:** LPLD **regressed** performance by −0.0975 mAP@0.5 (−13.6%). When the generator model is already at ≥0.70 mAP@0.5, only 167 of 781 LC boxes were valid uncertain predictions — the rest were noise. LPLD is only beneficial when the generator has low recall (many false negatives remaining). This defines a practical **boundary condition for LPLD application** not reported in the original paper (Yoon et al., 2024).

---

## 5.5 PyTorch Compatibility Patch (Critical)

PyTorch 2.6 changed `torch.load()` default to `weights_only=True`, breaking all Ultralytics `.pt` file loading. Every notebook cell that loads a `.pt` model must apply this patch:

```python
import ultralytics.nn.tasks as tasks
from ultralytics.utils.downloads import attempt_download_asset

def patched_torch_safe_load(weight):
    file = attempt_download_asset(weight)
    ckpt = torch.load(file, map_location='cpu', weights_only=False)
    return ckpt, file

tasks.torch_safe_load = patched_torch_safe_load
```

---

*Previous: [Chapter 4 — Dataset Preparation](./Chapter_4_Dataset_Preparation.md)*
