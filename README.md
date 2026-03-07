# MisterMobile - Carousell Image Cropper (Local/Windows)

Automated product photo processing for Carousell listings.

This README is aligned with `data-carocrop-local/processor.py` as currently configured.

## Current Behavior

- AI-first object detection and smart square crop (with CV fallback)
- Output is always `1080x1080` PNG
- Watermark is applied from `MM Watermark.png`
- Backdrop normalization code exists, but is disabled by default:
  - `ENABLE_BACKDROP_NORMALIZATION = False`
- Brightness is handled as full-frame enhancement with adaptive targeting:
  - `GLOBAL_BRIGHTNESS = 1.16`
  - `ENABLE_ADAPTIVE_BRIGHTNESS_TARGET = True`
  - Target luma and clamp limits are applied per image
- Anti-gray correction is enabled:
  - White-point lift (Lab L channel)
  - Mild color boost

## Processing Pipeline

Load -> AI detection (U2-Net) -> component filtering -> platform detection
-> logo check -> square crop -> border cleanup
-> optional localized backdrop normalization (off by default)
-> global enhancement (brightness/contrast/sharpness)
-> anti-gray correction (white-point + color)
-> resize 1080x1080 -> watermark -> save PNG

## Brightness and Color Tuning

Primary controls (top of `processor.py`):

- `ENABLE_BACKDROP_NORMALIZATION` (bool)
- `GLOBAL_BRIGHTNESS` (float)
- `ENABLE_ADAPTIVE_BRIGHTNESS_TARGET` (bool)
- `ADAPTIVE_BRIGHTNESS_TARGET_LUMA` (float)
- `ADAPTIVE_BRIGHTNESS_STRENGTH` (float)
- `ADAPTIVE_BRIGHTNESS_MIN` / `ADAPTIVE_BRIGHTNESS_MAX` (float clamp)
- `ENABLE_ANTI_GRAY_CORRECTION` (bool)
- `WHITE_POINT_PERCENTILE` (float)
- `WHITE_POINT_TARGET_LUMA` (float)
- `WHITE_POINT_MAX_GAIN` (float)
- `GLOBAL_COLOR_BOOST` (float)

Quick guidance:

- Too dark: increase `ADAPTIVE_BRIGHTNESS_TARGET_LUMA` first
- Too flat or gray: increase `WHITE_POINT_TARGET_LUMA` slightly, then `GLOBAL_COLOR_BOOST`
- Too bright/washed: reduce `GLOBAL_BRIGHTNESS` or lower `ADAPTIVE_BRIGHTNESS_MAX`

## Run and Build (Windows)

Python version is pinned to `3.12` for runtime stability with the compiled build.
`build.bat` and `run_local.bat` will stop with an error if the venv is not 3.12.

```powershell
# Create/recreate 3.12 venv
py -3.12 -m venv venv

# Install dependencies
.\venv\Scripts\python.exe -m pip install -r requirements.txt

# Run locally
run_local.bat
# or
.\venv\Scripts\python.exe processor.py

# Build stable app folder
build.bat

# Optional: experimental onefile build
build_onefile_experimental.bat

# Optional: PyInstaller fallback build
build_pyinstaller.bat
```

Build output:

```text
dist/
  MisterMobileCropper-v3.dist/
    MisterMobileCropper-v3.exe
    MM Watermark.png
    INSTRUCTIONS.html
    .u2net/
      u2net.onnx
    input/
    output/
```

`build.bat` now targets a stable standalone build.
`build_onefile_experimental.bat` is kept only for onefile testing.

PyInstaller fallback output:

```text
dist/
  pyinstaller/
    MisterMobileCropper-v3/
      MisterMobileCropper-v3.exe
      MM Watermark.png
      INSTRUCTIONS.html
      .u2net/
        u2net.onnx
      input/
      output/
```

## Output Specification

- Format: PNG
- Size: `1080x1080`
- Aspect ratio: `1:1`
- Filename: `{original_name}.png`

## Folder Layout

```text
data-carocrop-local/
  processor.py
  build.bat
  build_onefile_experimental.bat
  build_pyinstaller.bat
  run_local.bat
  requirements.txt
  MM Watermark.png
  INSTRUCTIONS.html
  input/
  output/
  dist/
  README.md
```

## Note About macOS Folder

`data-carocrop-mac/processor.py` is now synced with this local version.
