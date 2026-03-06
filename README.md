# Carousell Image Cropper (Local)

Automated image processing for Carousell listings with AI-first object detection, white-platform constrained crop, and 1080×1080 output.

## Key Features

1. **AI-First Detection** — U2-Net (`rembg`) with mask dilation for robust edge detection.
2. **Lab-Based CV Fallback** — Lab L-channel + Otsu thresholding + morphology when AI fails.
3. **Mask Dilation** — 3px elliptical dilation fuses fragile device edges before contour extraction, preventing wrong-object detection.
4. **Light-Box Strategy** — Border margin exclusion (5% extreme, 8% side) with center-biased contour scoring.
5. **White Platform Constraint** — Crop is constrained to the detected white platform area.
6. **Yellow Logo Inclusion** — Yellow mascot/logo is always included in crop bounds when detected.
7. **Artifact-Aware Padding** — 6% padding around detected product, with per-side contamination probes that halve padding if artifacts (paper, backdrop) are detected in the outer zone.
8. **Smart Square Crop** — Final output is always 1:1 (1080×1080) with priority: full product in frame > maximum coverage.
9. **Adaptive Enhancement** — Dynamic brightness/contrast/sharpness based on image analysis.
10. **Backdrop Brightening** — Selective backdrop brightening while preserving product colors.
11. **Padding-Preserving Border Cleanup** — Post-crop symmetrical trimming capped at 2% to protect padding.
12. **Comparison Mode** — Side-by-side input/output images for visual QC (`--compare`).
13. **Recursive Scanning** — Processes all supported images in subfolders.
14. **HEIC Support** — iPhone HEIC/HEIF support via `pillow-heif`.
15. **MM Watermark** — Watermark asset included (`MM Watermark.png`) for future implementation.

## Processing Pipeline

```text
Load → AI Detection (U2-Net + mask dilation) → [Lab-based CV Fallback]
→ Filter AI Components → White Platform Detection → Logo Merge
→ Artifact-Aware Padding → Square Crop Solve → Border Cleanup (2% cap)
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
├── input/           # Raw photos (JPG, JPEG, PNG, WebP, HEIC, HEIF) [gitignored]
├── output/          # Processed outputs (1080×1080 PNG) [gitignored]
├── comparisons/     # Side-by-side QC comparisons [gitignored]
├── processor.py     # Main processing pipeline
├── run_local.bat    # Local batch launcher (with --compare)
├── requirements.txt # Python dependencies
├── MM Watermark.png # Watermark asset (for future use)
├── .gitignore       # Git ignore rules
└── README.md        # This file
```

## Tunable Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `TARGET_SIZE` | 1080 | Output dimensions (1080×1080) |
| `DEFAULT_PADDING_PCT` | 0.06 (6%) | Breathing room around detected product |
| `AI_MASK_DILATE_PX` | 3 | Dilation kernel radius for AI mask |
| `PADDING_ARTIFACT_THRESHOLD` | 0.50 | Contamination score before padding is halved |
| `PLATFORM_X_TRIM_PCT` | 0.05 | X-margin trimmed from white platform |
| `PLATFORM_Y_TRIM_PCT` | 0.03 | Y-margin trimmed from white platform |

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
- **6% padding** — visible breathing room on all sides (halved if artifacts detected).
- **Edge artifacts always trimmed** — even if they clip the product edges.
- **Yellow logo included** — always merged into required bounds when detected.
- **Minimum zoom floor** — crop is at least 50% of the image's shorter dimension.

## Logging Tags

| Tag | Description |
|-----|-------------|
| `[AI-FIRST]` | AI detection and CV fallback status |
| `[AI-COMPONENT]` | Component filtering (kept/dropped with scores) |
| `[PLATFORM]` | Detected white-platform bounds |
| `[LOGO]` | Yellow mascot/logo detection |
| `[PAD-GUARD]` | Per-side artifact probe results and padding adjustments |
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
