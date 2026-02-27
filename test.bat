@echo off
echo ============================================
echo   Testing Carousell Image Cropper
echo ============================================
echo.

cd /d "%~dp0"
echo Running processor with first 3 images only...
echo.

.\venv\Scripts\python -c "import sys; sys.path.insert(0, '.'); from processor import *; from pathlib import Path; input_dir = Path('input'); images = list(input_dir.glob('*.jpg'))[:3] + list(input_dir.glob('*.JPG'))[:3] + list(input_dir.glob('*.png'))[:3]; output_dir = Path('output'); output_dir.mkdir(exist_ok=True); [process_image(img, output_dir) for img in images[:3]]"

echo.
echo ============================================
echo   Test Complete!
echo ============================================
pause
