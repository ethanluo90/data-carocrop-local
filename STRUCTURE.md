# Carousell Image Cropper - Project Structure

## Directory Overview

```
data-carocrop/
├── input/              # Raw product photos (place images here)
├── output/             # Processed 1080x1080 images
├── comparisons/        # Before/after validation images
├── venv/               # Python virtual environment
├── __pycache__/        # Python cache files
│
├── processor.py        # Main image processing script
├── image_qc.py         # Quality control & centering validation
├── requirements.txt    # Python dependencies
├── run.bat             # Batch processing launcher
│
├── README.md           # User documentation
├── TECHNICAL.md        # Technical algorithm documentation
└── STRUCTURE.md        # This file
```

## File Descriptions

### Core Scripts

- **processor.py** (15.6 KB)
  - Main processing pipeline
  - Edge detection & contour finding
  - Safe tight crop algorithm
  - Image enhancement
  - Batch processing with progress bar

- **image_qc.py** (6.0 KB)
  - Quality control validation
  - Before/after comparison generation
  - Centering accuracy checking
  - Off-center image detection

### Configuration

- **requirements.txt**
  - Python package dependencies
  - pillow, pillow-heif, numpy, opencv-python

- **run.bat**
  - Windows batch launcher
  - Activates venv and runs processor.py

### Documentation

- **README.md**
  - User-facing documentation
  - Quick start guide
  - Feature overview
  - Usage instructions

- **TECHNICAL.md**
  - Algorithm documentation
  - Step-by-step technical details
  - Performance characteristics
  - Configuration parameters

- **STRUCTURE.md**
  - Project structure overview
  - File descriptions
  - Workflow documentation

## Workflow

### 1. Input
- Place raw photos in `input/` folder
- Supported formats: JPG, PNG, HEIC, WebP

### 2. Processing
- Run `run.bat` or `python processor.py`
- Pipeline: Load → Detect → Crop → Enhance → Save
- Progress bar shows real-time status

### 3. Output
- Processed images saved to `output/`
- Format: PNG, 1080x1080, named `{original}_caro.png`

### 4. Validation
- Run `python image_qc.py` for quality control
- Generates side-by-side comparisons in `comparisons/`
- Reports centering accuracy

## Key Features

### Edge Detection
- Uses OpenCV Canny edge detection
- Automatically finds product outline
- No manual configuration needed

### Safe Tight Crop
- Crops as tight as possible without cutting product
- 2% safety margin prevents edge cutting
- Automatic boundary adjustment

### Image Enhancement
- Brightness: +8%
- Contrast: +5%
- Sharpness: +30%

## Performance

- **Processing Speed**: 0.3-0.5s per image
- **Success Rate**: 100% on clean backdrop photos
- **Batch Processing**: 33 images in ~17 seconds

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pillow | ≥10.0.0 | Image processing |
| pillow-heif | ≥1.1.0 | HEIC support |
| numpy | ≥1.24.0 | Array operations |
| opencv-python | ≥4.8.0 | Edge detection |

## Best Practices

### Input Images
- ✅ Light/white backdrop
- ✅ Dark products (phones, tablets, boxes)
- ✅ Good lighting
- ✅ Product centered-ish in frame

### Output Quality
- All products fill 85-95% of frame
- No product edges cut
- Perfect 1:1 square aspect ratio
- Enhanced for marketplace listings
