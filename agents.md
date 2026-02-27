# Carousell Image Cropper - Technical Documentation

## Overview

Automated image processing for Carousell marketplace listings. Crops product photos to 1080x1080 square format with centered products and seamless background blending.

## Processing Pipeline

```
Input → Sample Backdrop Color → Smart Crop → Resize 1080x1080 → Output
```

## Key Features

| Feature | Description |
|---------|-------------|
| Backdrop Color Sampling | Matches canvas to actual backdrop color |
| Object Detection | Center-region centroid detection |
| Smart Crop | Object-centered square crop |
| Seamless Blending | No visible edge artifacts |
| Progress Bar | Visual progress with timing |

## File Structure

```
data-carocrop/
├── input/              # Raw photos
├── output/             # Processed PNGs (1080x1080)
├── comparisons/        # Before/after images
├── processor.py        # Main script
├── image_qc.py         # QC script
├── run.bat             # Process batch file
└── requirements.txt    # Dependencies
```

## Dependencies

| Package | What It Does |
|---------|--------------|
| **pillow** | Image processing (load, save, resize, crop) |
| **pillow-heif** | HEIC/HEIF format support for iPhone photos |
| **numpy** | Fast array operations for color sampling |

## Scripts

### processor.py Functions

| Function | Purpose |
|----------|---------|
| `load_image()` | Load with HEIC support + EXIF handling |
| `sample_backdrop_color()` | Sample median light color from edges |
| `smart_crop_to_content()` | Centroid-based object detection & crop |
| `process_image()` | Full pipeline with timing |

## Algorithm Details

### Backdrop Color Sampling

1. Examine outer 15% margin of image
2. Filter to light pixels (brightness > 150)
3. Calculate median RGB color
4. Use this color for canvas instead of white

### Object Detection

1. Examine center 60% of image
2. Find pixels below brightness threshold (180)
3. Calculate centroid (center of mass)
4. Use centroid as crop center

### Crop Sizing

1. Object dimensions + 5% padding
2. Cap at 80% of image smaller dimension
3. Minimum 800px
4. Paste onto sampled-color canvas
5. Resize to 1080x1080

## Output Specs

| Property | Value |
|----------|-------|
| Format | PNG (RGB) |
| Size | 1080 x 1080 px |
| Aspect | 1:1 Square |
| Background | Sampled backdrop color |
| Quality | 95 |

## Changelog

### 2026-01-23
- Implemented backdrop color sampling for seamless blending
- Removed artificial whitening effects
- Canvas now matches actual backdrop color
- Tightened cropping to 5% padding

### 2026-01-22
- Added centroid-based object detection
- Created comparisons folder and image_qc.py
- Fixed centering issues

### 2026-01-21
- Initial implementation
- Smart crop with white canvas
- Progress bar with timing
