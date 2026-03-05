@echo off
echo ============================================
echo   Carousell Image Cropper
echo   Format: 1080x1080px Square
echo ============================================
echo.

cd /d "%~dp0"
echo Press any key to start processing...
pause >nul
echo.
echo Running cropper...
echo.

.\venv\Scripts\python processor.py --compare

echo.
echo ============================================
echo   Processing Complete!
echo ============================================
echo.
echo Output folder: output\
echo Comparisons:   comparisons\
echo.
pause
