# DocQuadNet-256 – Integration Package for OSS Document Scanner

This package contains all components needed to integrate the DocQuadNet-256 model
from **MakeACopy** into **OSS Document Scanner** (or any other Android app).

---

## Contents

```
docquad-integration/
├── README.md                          ← This file
├── model/
│   ├── docquadnet256_trained_opset17.ort                    ← ONNX model (ORT format)
│   └── docquadnet256_trained_opset17.required_operators.config  ← Required ONNX ops
└── src/docquad/
    ├── DocQuadLetterbox.java          ← Letterbox transformation (platform-independent)
    ├── DocQuadScore.java              ← Geometry utilities (platform-independent)
    ├── DocQuadPostprocessor.java      ← Postprocessing pipeline (platform-independent)
    ├── DocQuadOrtRunner.java          ← ONNX Runtime wrapper (Android)
    └── DocQuadDetector.java           ← High-level detector (Android)
```

---

## Model Specification

| Property | Value |
|---|---|
| **Format** | ONNX Runtime Mobile (`.ort`, optimized for Android) |
| **ONNX Opset** | 17 |
| **Input Name** | `input` |
| **Input Shape** | `[1, 3, 256, 256]` — Batch=1, RGB, NCHW, float32, values 0..1 |
| **Output 1** | `corner_heatmaps` `[1, 4, 64, 64]` — Heatmaps for TL, TR, BR, BL |
| **Output 2** | `mask_logits` `[1, 1, 64, 64]` — Document segmentation mask (logits) |
| **Required Ops** | Conv, FusedConv, Resize, Mul, GlobalAveragePool, HardSigmoid |

### Corner Order

The 4 channels in `corner_heatmaps` correspond to:
- Channel 0: **Top-Left** (TL)
- Channel 1: **Top-Right** (TR)
- Channel 2: **Bottom-Right** (BR)
- Channel 3: **Bottom-Left** (BL)

---

## Architecture Overview

The pipeline consists of 4 stages:

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│ Input Image │────▶│  Letterbox   │────▶│ ONNX Inference│────▶│ Postprocessing   │
│ (any size)  │     │  → 256×256   │     │ DocQuadNet   │     │ → 4 corners (orig)│
└─────────────┘     └──────────────┘     └──────────────┘     └──────────────────┘
```

### Stage 1: Letterboxing (`DocQuadLetterbox`)

The input image is scaled to 256×256 while preserving the aspect ratio and centered.
Unfilled areas are padded with black (0,0,0).

```java
DocQuadLetterbox lb = DocQuadLetterbox.create(srcWidth, srcHeight);
// lb.scale   → Scale factor
// lb.offsetX → Horizontal offset in 256-space
// lb.offsetY → Vertical offset in 256-space
```

### Stage 2: Bitmap → NCHW float32

The 256×256 bitmap is converted into a `float[3 * 256 * 256]` array:
- Layout: **NCHW** (channel-first)
- Channels: **R, G, B** (in this order)
- Value range: **0.0 .. 1.0** (pixel / 255.0)

```java
// Pseudocode:
float[] input = new float[3 * 256 * 256];
for (y, x in 256×256):
    input[y*256+x]             = R(x,y) / 255.0f;  // Channel 0 (Red)
    input[65536 + y*256+x]     = G(x,y) / 255.0f;  // Channel 1 (Green)
    input[2*65536 + y*256+x]   = B(x,y) / 255.0f;  // Channel 2 (Blue)
```

### Stage 3: ONNX Inference (`DocQuadOrtRunner`)

```java
DocQuadOrtRunner runner = DocQuadOrtRunner.getInstance(context, "docquad/docquadnet256_trained_opset17.ort");
DocQuadOrtRunner.Outputs outputs = runner.run(inputNchw);
float[][][][] cornerHeatmaps = outputs.cornerHeatmaps();  // [1,4,64,64]
float[][][][] maskLogits     = outputs.maskLogits();       // [1,1,64,64]
```

**Execution Providers** (fallback chain): NNAPI → XNNPACK → CPU

### Stage 4: Postprocessing (`DocQuadPostprocessor`)

Two paths are evaluated in parallel:

#### Path A: Corner Heatmaps (primary path)
1. **Argmax** per channel → peak position in 64×64 space
2. **Mapping 64→256**: `x256 = (ix + 0.5) * 4.0`
3. Optional: **3×3 subpixel refinement** (weighted centroid around peak)

#### Path B: Mask → Quad (secondary path)
1. Binary mask: `sigmoid(logit) > 0.5`
2. **PCA** on mask pixels → principal axes
3. **Oriented bounding rectangle** from u/v extrema
4. **Canonical order**: angle sorting + TL rotation

#### Decision (Path Choice)
A penalty-based algorithm selects the better quad:
- **Geometry penalty**: OOB, self-intersection, convexity, area, edge ratio
- **Mask disagreement**: comparison of corner quad vs. mask at 8×8 sample points
- **Guardrails**: agreement check (max 32px deviation), score margin (50.0)

```java
DocQuadPostprocessor.Result r = DocQuadPostprocessor.postprocess(
    cornerHeatmaps, maskLogits, lb, DocQuadPostprocessor.PeakMode.REFINE_3X3);

double[][] cornersOriginal = r.chosenQuadOriginal;
// cornersOriginal[0] = {xTL, yTL}
// cornersOriginal[1] = {xTR, yTR}
// cornersOriginal[2] = {xBR, yBR}
// cornersOriginal[3] = {xBL, yBL}
```

---

## Quick Start: Integration in 5 Steps

### Step 1: Add dependency

In `build.gradle` (app):

```groovy
dependencies {
    implementation 'com.microsoft.onnxruntime:onnxruntime-android:latest.release'
}
```

### Step 2: Copy model to assets

```
app/src/main/assets/docquad/docquadnet256_trained_opset17.ort
```

Copy the file from `model/` to this path.

### Step 3: Copy Java sources

Copy all `.java` files from `src/docquad/` into your project.
Adjust the `package` name to match your project structure, e.g.:

```
com.example.ossdocscanner.ml.docquad
```

### Step 4: Use the high-level API

```java
// Once (e.g., at app start or fragment init):
DocQuadOrtRunner.getInstanceAsync(context, DocQuadDetector.DEFAULT_MODEL_ASSET_PATH);

// For each image:
DocQuadDetector detector = new DocQuadDetector();
double[][] corners = detector.detect(bitmap, context);

if (corners != null) {
    // corners[0] = Top-Left     {x, y}
    // corners[1] = Top-Right    {x, y}
    // corners[2] = Bottom-Right {x, y}
    // corners[3] = Bottom-Left  {x, y}
    // Coordinates are in original bitmap pixels
}
```

### Step 5: Release resources

```java
// When no longer needed (e.g., onDestroy):
DocQuadOrtRunner.releaseInstance();
```

---

## Minimal Integration (without Dual-Path)

If you only want to use the corner heatmaps (simpler, ~50 lines):

```java
// 1. Letterbox
DocQuadLetterbox lb = DocQuadLetterbox.create(srcW, srcH);

// 2. Bitmap → 256×256 → NCHW float
// (see bitmapToNchwFloat01 in DocQuadDetector.java)

// 3. Inference
DocQuadOrtRunner runner = DocQuadOrtRunner.getInstance(ctx, modelPath);
DocQuadOrtRunner.Outputs out = runner.run(inputNchw);

// 4. Argmax → corners
double[][] corners256 = DocQuadPostprocessor.argmaxCorners64ToCorners256(out.cornerHeatmaps());

// 5. Back-transformation
double[][] cornersOrig = DocQuadPostprocessor.mapCorners256ToOriginal(corners256, lb);
```

This requires `DocQuadLetterbox`, `DocQuadPostprocessor` (only `argmaxCorners64ToCorners256`
and `mapCorners256ToOriginal`), and `DocQuadOrtRunner`.
`DocQuadScore` and `DocQuadDetector` can be omitted.

---

## Kotlin Porting

The classes `DocQuadLetterbox`, `DocQuadScore`, and `DocQuadPostprocessor` have
**no Android dependencies** and can be ported 1:1 to Kotlin.
All logic is purely mathematical (`double[][]`, `float[][][][]`, `Math.*`).

Only `DocQuadOrtRunner` and `DocQuadDetector` use Android APIs:
- `Context`, `AssetManager` → for model loading
- `Bitmap`, `Canvas` → for letterbox rendering
- `Log` → for logging

---

## Platform Independence of Components

| Class | Android-free? | Description |
|---|---|---|
| `DocQuadLetterbox` | ✅ Yes | Letterbox calculation (pure math) |
| `DocQuadScore` | ✅ Yes | Geometry validation (pure math) |
| `DocQuadPostprocessor` | ✅ Yes | Postprocessing pipeline (pure math) |
| `DocQuadOrtRunner` | ❌ No | ONNX Runtime + Android Context/Log |
| `DocQuadDetector` | ❌ No | Bitmap/Canvas + Context |

---

## Performance Notes

- **Model loading**: ~100–300ms (one-time). Use `getInstanceAsync()` for non-blocking loading.
- **Inference**: ~10–50ms per image (depending on device and execution provider).
- **Postprocessing**: <1ms (pure CPU computation).
- **Memory**: The model is copied to cache and loaded via mmap (memory-efficient).
- **Bitmap recycling**: `DocQuadDetector` automatically recycles the internal 256×256 bitmap.

---

## License

The DocQuadNet-256 model and associated code originate from the
[MakeACopy](https://github.com/egdels/makeacopy) project and are subject to
the license specified there (Apache License 2.0). Please verify compatibility
with the OSS Document Scanner license before integration.
