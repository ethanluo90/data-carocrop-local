# Carousell Image Cropper (Local)

Automated image processing for Carousell listings with AI-first object detection, white-platform constrained crop, and 1080x1080 output.

## Locked Rules and Features

1. **AI-First Detection** - U2-Net (`rembg`) is the primary detector for product bounds.
2. **White Backdrop Only** - Crop is constrained to detected white platform area and trimmed away from gray side edges.
3. **Yellow Logo Inclusion (Compulsory)** - If yellow mascot/logo is detected, it is always included in crop bounds.
4. **Smart Square Crop** - Final output is always 1:1 (1080x1080).
5. **Priority-Based Cropping**
   - Priority 1: Full product in frame (never cut off if geometrically possible).
   - Priority 2: Maximum coverage (smallest valid square containing required bounds).
6. **Adaptive Enhancement** - Dynamic brightness/contrast/sharpness based on image analysis.
7. **Backdrop Brightening** - Selective backdrop brightening while preserving product colors.
8. **Recursive Scanning** - Processes all supported images in subfolders.
9. **HEIC Support** - iPhone HEIC/HEIF support via `pillow-heif`.

## Processing Pipeline

```text
Load -> AI-First Detection -> (CV Fallback if needed) -> White Platform Detection -> Compulsory Logo Merge -> Priority Square Solve -> Crop -> Backdrop Brighten -> Adaptive Enhance -> Resize 1080x1080 -> Save
```

## Quick Start

```powershell
# From this folder
run_local.bat
```

## Folder Structure

```text
data-carocrop-local/
|-- input/          # Raw photos (JPG, JPEG, PNG, WebP, HEIC, HEIF)
|-- output/         # Processed outputs (1080x1080 PNG)
|-- comparisons/    # Optional QC comparisons
|-- processor.py    # Main processing pipeline
|-- image_qc.py     # Quality control helper
|-- run_local.bat   # Local batch launcher
|-- requirements.txt
```

## CLI Usage

```powershell
python processor.py --input "C:\path\to\input" --output "C:\path\to\output"
```

- `--input` and `--output` are optional.
- If omitted, defaults are `./input` and `./output`.

## Output Specification

- Format: `PNG`
- Size: `1080x1080`
- Aspect Ratio: `1:1`
- Filename: `{original}_caro.png`

## Logging Tags

`processor.py` emits structured tags for troubleshooting:

- `[AI-FIRST]` - AI detection and CV fallback status
- `[PLATFORM]` - detected white-platform bounds
- `[LOGO]` - yellow mascot/logo detection
- `[CROP-SOLVER]` - crop solve decisions
- `[CONSTRAINT-CONFLICT]` - conflict between required bounds and platform constraints

## Installation

```powershell
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
```

## Dependencies

- `pillow`
- `pillow-heif`
- `numpy`
- `opencv-python`
- `rembg`
- `onnxruntime`

## Notes

- If product+logo cannot fully fit in platform-constrained area, solver falls back to full-image square constraints and logs `[CONSTRAINT-CONFLICT]`.
- Quality checks can be run separately with `image_qc.py`.
