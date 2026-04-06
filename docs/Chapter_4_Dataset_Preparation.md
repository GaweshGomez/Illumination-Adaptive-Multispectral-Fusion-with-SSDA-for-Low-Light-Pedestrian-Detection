# Chapter 4: Dataset Preparation

> **Module:** CM4605 Individual Research Project  
> **Student:** Gawesh Gomes · RUG ID: 2313082  
> **Project:** Adaptive Multispectral Fusion for Low-Light Pedestrian Detection with Semi-Supervised Domain Adaptation  
> **Platform:** Kaggle Notebooks · Tesla T4 GPU

---

## 4.1 Overview

This project uses two distinct datasets across its two experimental phases. **LLVIP** serves as the labelled source domain for Part A (fusion model training), and **CVC-14 Night** serves as the unlabelled-heavy target domain for Part B (semi-supervised domain adaptation). Both datasets are nighttime pedestrian detection benchmarks, which is a deliberate design choice to minimise photometric domain gap while studying geometric domain shift.

---

## 4.2 Source Domain: LLVIP Dataset

### 4.2.1 Description

| Property | Value |
|---|---|
| Full Name | Low-Light Visible and Infrared Paired Dataset |
| Location | Beijing, China |
| Scene Type | Nighttime urban pedestrian surveillance (fixed camera) |
| Training Images | 12,025 paired RGB + Thermal frames |
| Validation Images | 3,463 paired RGB + Thermal frames |
| Class | Single class — `person` (class 0) |
| Image Size | 640 × 640 (after preprocessing) |
| Annotation Format | YOLO normalised bounding box |
| RGB Mean Brightness | 0.1226 (extremely dark, nighttime) |
| Kaggle Path | `kaggle/input/llvip-rgb-thermal-yolo-format/processed` |

### 4.2.2 Key Properties

- RGB and thermal channels are **perfectly aligned** — captured at the same timestamp with co-registered sensors.
- Images are predominantly thermal-dominant at night (RGB weight mean = 0.4263), which validates the illumination-adaptive fusion design.
- Standard deviation of per-image RGB weight = **0.0908**, confirming genuine per-image variation (Simple Fusion would have std = 0.000).

### 4.2.3 Preprocessing Steps

```
1. Extract RGB channel from paired LLVIP images
2. Extract FIR thermal channel from paired LLVIP images
3. Compute per-image illumination scalar:
       illumination = mean(RGB_float)        # scalar in [0, 1]
       rgb_w = 1 / (1 + exp(-(illumination * 4.0 - 1.0)))
4. Generate fused image:
       fused = rgb_w * RGB_float + (1 - rgb_w) * Thermal_float
       fused_uint8 = clip(fused * 255, 0, 255).astype(uint8)
5. Write YOLO-format data.yaml pointing to fused images
6. Convert labels to YOLO normalised centre format (already in source)
```

### 4.2.4 LLVIP Fusion Statistics

| Statistic | Value |
|---|---|
| RGB weight (rgb_w) min | 0.3165 |
| RGB weight (rgb_w) max | 0.7676 |
| RGB weight (rgb_w) mean | 0.4263 |
| RGB weight (rgb_w) std | 0.0908 |

> **Interpretation:** Mean rgb_w = 0.4263 means images are on average **57.4% thermal, 42.6% RGB** — correct for nighttime scenes. The non-zero std proves genuine per-image adaptivity.

---

## 4.3 Target Domain: CVC-14 Night Dataset

### 4.3.1 Description

| Property | Value |
|---|---|
| Full Name | CVC-14 Night Pedestrian Dataset |
| Location | Barcelona, Spain |
| Scene Type | Nighttime pedestrian detection from a moving bus (street-level) |
| Class | Single class — pedestrian (class 0) |
| Image Size | **640 × 471** (NOT 640 × 512 as cited in most papers) |
| Annotation Format | 11-column non-standard format (see §4.3.3) |
| Kaggle Path | `kaggle/input/cvc14-night-rgb-fir/Night` |

> **Critical Discovery:** CVC-14 images are **640 × 471**, not 640 × 512. Using 512 as the height would miscalculate all y-coordinates by 8 pixels. The annotation converter explicitly uses `H=471, W=640`.

### 4.3.2 Dataset Split Structure

| Split | Folder | Count | Labels | Purpose |
|---|---|---|---|---|
| FramesPos (GT) | `FramesPos/` | 614 images | Non-empty GT annotation files | Supervised training seed |
| FramesPos (Unlabelled) | `FramesPos/` | 1,972 images | No annotation (Pool A) | Pseudo-label mining pool |
| FramesNeg | `FramesNeg/` | 802 images | Empty label files (confirmed background) | Hard negatives throughout training |
| NewTest | `NewTest/` | 727 images | 1,894 GT bounding box instances | **Held-out evaluation set** |
| **Total Training Pool** | — | **3,388 images** | — | — |

> **Pool A Note:** CVC-14 authors left 1,972 FramesPos images unannotated due to crowding, extreme scale, or occlusion. These images DO contain pedestrians and form the pseudo-labelling target pool for SSDA.

> **NewTest Note:** Images are from a **different recording session** (different date/time) than training data, guaranteeing no temporal overlap. All primary mAP@0.5 metrics are evaluated on NewTest only.

### 4.3.3 Annotation Format Conversion

The original CVC-14 annotation files use an 11-column non-standard format:

```
xmin  ymin  width  height  flag  0  0  0  0  obj_id  0
```

Conversion to YOLO normalised centre format:

```python
cx = (xmin + width/2)  / 640   # W = 640
cy = (ymin + height/2) / 471   # H = 471  ← critical: not 512
wn = width  / 640
hn = height / 471
```

### 4.3.4 CVC-14 Fusion Statistics

The same sigmoid illumination-adaptive formula used for LLVIP is applied identically to CVC-14 paired images. This ensures CVC-14 fused images are in the **same statistical format** as the LLVIP images the source model was trained on.

| Statistic | LLVIP | CVC-14 Night |
|---|---|---|
| rgb_w min | 0.3165 | 0.3385 |
| rgb_w max | 0.7676 | 0.7919 |
| rgb_w mean | 0.4263 | 0.4900 |
| rgb_w std | 0.0908 | 0.0636 |

> Both datasets are thermal-dominant at night (both means below 0.50), validating the fusion strategy for cross-dataset use.

---

## 4.4 Label Usage Summary Across Experiments

| Experiment Phase | Labels Used | Label % of Total Pool | Source |
|---|---|---|---|
| Part A — All Fusion Ablations | 15,104 LLVIP labels | 100% | LLVIP training set |
| Part B V1 — Warm-Start / SSDA R1 | 114 GT (CVC-14) | 3.4% (114 / 3,388) | FramesPos subset |
| Part B V2 — Phase 2, 3A, 3B | 614 GT (CVC-14) | 18.1% (614 / 3,388) | All available FramesPos GT |

> The **V1 experiments** artificially withhold 500 available GT images to simulate an extreme low-label scenario (only 20% of available GT used). The **V2 experiments** use all available GT labels, representing the realistic deployment scenario.

---

## 4.5 Negative Result: FLIR ADAS Dataset (Abandoned)

Before CVC-14 was selected, an initial attempt was made to adapt the LLVIP-trained model to the **FLIR ADAS v1.3** thermal dataset. This failed completely for the following reasons:

| Failure Reason | Detail |
|---|---|
| Near-total domain collapse | mAP@0.5 stayed near 0.00 even after warm-start |
| Wrong scene type | LLVIP = surveillance; FLIR = dashcam (different geometry) |
| Multi-class complexity | FLIR labels cars, trucks, bicycles, people |
| Sensor mismatch | Different noise pattern, resolution, temperature range encoding |
| Annotation format | COCO-style JSON requiring significant preprocessing overhead |

CVC-14 Night was selected as the replacement because it shares the single-class nighttime pedestrian surveillance setting with LLVIP and provides the paired-modality annotations necessary for SSDA.

---

## 4.6 Domain Gap Analysis

### Histogram Matching Experiment (Negative Result)

Histogram matching was applied to align CVC-14 fused image statistics (mean = 104.4) to LLVIP statistics (mean = 51.8). Result: **no improvement** in zero-shot mAP@0.5. The domain gap is **not photometric**.

### Centre-Offset Diagnostic (Positive Result)

For each zero-shot prediction, the distance from the predicted box centre to the nearest GT box centre was measured.

| Metric | Value |
|---|---|
| Mean centre offset | 72.9 pixels |
| As % of image width | ~11% |

> **Interpretation:** The source model knows *where* pedestrians are, but predicts wrong box shapes and scales. CVC-14 pedestrians appear smaller and at different aspect ratios than LLVIP pedestrians. The domain gap is **geometric (structural)**, not photometric.

---

*Next: [Chapter 5 — System Design](./Chapter_5_System_Design.md)*
