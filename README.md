# Carousell Image Cropper (Local)

Automated image processing for Carousell listings with AI-first object detection, white-platform constrained crop, and 1080×1080 output.

## Key Features

1. **AI-First Detection** — U2-Net (`rembg`) is the primary detector for product bounds.
2. **Lab-Based CV Fallback** — Lab L-channel + Otsu thresholding + morphology for robust segmentation when AI fails.
3. **Light-Box Strategy** — Border margin exclusion (5% extreme, 8% side) with center-biased contour scoring to prevent edge artifacts from light-box seams/panels.
4. **White Platform Constraint** — Crop is constrained to the detected white platform area, trimmed away from gray side edges.
5. **Yellow Logo Inclusion** — If yellow mascot/logo is detected, it is always included in crop bounds.
6. **Smart Square Crop** — Final output is always 1:1 (1080×1080) with priority: full product in frame > maximum coverage.
7. **Adaptive Enhancement** — Dynamic brightness/contrast/sharpness based on image analysis.
8. **Backdrop Brightening** — Selective backdrop brightening while preserving product colors.
9. **Border Cleanup** — Post-crop symmetrical trimming of residual dark/contaminated edges.
10. **Comparison Mode** — Side-by-side input/output images for visual QC (`--compare`).
11. **Recursive Scanning** — Processes all supported images in subfolders.
12. **HEIC Support** — iPhone HEIC/HEIF support via `pillow-heif`.

## Processing Pipeline

```text
Load → AI Detection (U2-Net) → [Lab-based CV Fallback] → Filter AI Components
→ White Platform Detection → Logo Merge → Square Crop Solve → Border Cleanup
→ Backdrop Brighten → Adaptive Enhance → Resize 1080×1080 → Save
```

## Quick Start

```powershell
# From this folder
run_local.bat
```

## Folder Structure

```text
data-carocrop-local/
├── input/           # Raw photos (JPG, JPEG, PNG, WebP, HEIC, HEIF)
├── output/          # Processed outputs (1080×1080 PNG)
├── comparisons/     # Side-by-side QC comparisons (when --compare used)
├── processor.py     # Main processing pipeline
├── run_local.bat    # Local batch launcher (with --compare)
├── requirements.txt # Python dependencies
└── MM Watermark.png # Watermark asset
```

## CLI Usage

```powershell
python processor.py [--input PATH] [--output PATH] [--compare]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `./input` | Input folder with raw photos |
| `--output` | `./output` | Output folder for processed images |
| `--compare` | off | Generate side-by-side comparisons in `comparisons/` |

## Output Specification

| Property | Value |
|----------|-------|
| Format | PNG |
| Size | 1080×1080 |
| Aspect Ratio | 1:1 |
| Filename | `{original}_caro.png` |

## Cropping Rules

- **Full product in frame** — never cut off if geometrically possible.
- **Edge artifacts always trimmed** — even if they clip the product edges.
- **Yellow logo included** — always merged into required bounds when detected.
- **Minimum zoom floor** — crop is at least 50% of the image's shorter dimension to prevent over-zoom on low-contrast products.

## Logging Tags

| Tag | Description |
|-----|-------------|
| `[AI-FIRST]` | AI detection and CV fallback status |
| `[AI-COMPONENT]` | Component filtering (kept/dropped with scores) |
| `[PLATFORM]` | Detected white-platform bounds |
| `[LOGO]` | Yellow mascot/logo detection |
| `[CROP-SOLVER]` | Crop solve decisions and square sizing |
| `[BORDER-CLEANUP]` | Post-crop edge trimming results |
| `[CONSTRAINT-CONFLICT]` | Conflict between required bounds and platform |

## Installation

```powershell
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `pillow` | Image processing |
| `pillow-heif` | HEIC/HEIF support |
| `numpy` | Array operations |
| `opencv-python` | CV detection, Lab color space, morphology |
| `rembg` | AI saliency detection (U2-Net) |
| `onnxruntime` | ONNX model runtime for rembg |
