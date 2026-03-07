@echo off
REM ============================================
REM   Nuitka Build Script
REM   Compiles processor.py into a single-file executable
REM ============================================

cd /d "%~dp0"

set "REQ_PY=3.12"
if not exist ".\venv\Scripts\python.exe" (
    echo [ERROR] Missing venv Python at .\venv\Scripts\python.exe
    echo         Create it with:
    echo         py -3.12 -m venv venv
    pause
    exit /b 1
)

for /f %%v in ('.\venv\Scripts\python.exe -c "import sys; print(sys.version_info[0], sys.version_info[1], sep='.') "') do set "PY_VER=%%v"
if not "%PY_VER%"=="%REQ_PY%" (
    echo [ERROR] Python %REQ_PY% is required, but venv has %PY_VER%.
    echo         Recreate venv with:
    echo         rmdir /s /q venv
    echo         py -3.12 -m venv venv
    echo         .\venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo [1/4] Checking build dependencies...
.\venv\Scripts\python.exe -c "import nuitka, ordered_set, zstandard, numba, llvmlite" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo   Missing build deps. Installing...
    .\venv\Scripts\python.exe -m pip install nuitka ordered-set zstandard numba llvmlite
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo [ERROR] Failed to install build dependencies.
        echo         Check internet/firewall or install manually:
        echo         .\venv\Scripts\python.exe -m pip install nuitka ordered-set zstandard numba llvmlite
        pause
        exit /b 1
    )
) else (
    echo   Build deps already installed.
)

.\venv\Scripts\python.exe -c "import nuitka" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Nuitka is not available in this venv.
    pause
    exit /b 1
)

echo.
echo [2/4] Cleaning stale build artifacts...
if exist "dist\processor.build" rmdir /s /q "dist\processor.build"
if exist "dist\processor.dist" rmdir /s /q "dist\processor.dist"
if exist "dist\processor.onefile-build" rmdir /s /q "dist\processor.onefile-build"
if exist "dist\MisterMobileCropper-v3.dist" rmdir /s /q "dist\MisterMobileCropper-v3.dist"
if exist "dist\MisterMobileCropper-v3.exe" del /q "dist\MisterMobileCropper-v3.exe"
echo   Cleaned.

echo.
echo [3/4] Preparing external AI model...
if not exist ".u2net" mkdir ".u2net"
if exist "%USERPROFILE%\.u2net\u2net.onnx" (
    copy "%USERPROFILE%\.u2net\u2net.onnx" ".u2net\u2net.onnx" >nul
    echo   Model prepared: .u2net\u2net.onnx
) else (
    echo   [WARNING] u2net.onnx not found at %USERPROFILE%\.u2net\
    echo   The exe will download the model on first run ^(internet required^).
)

echo.
echo [4/4] Building onefile executable...
.\venv\Scripts\python.exe -m nuitka ^
    --onefile ^
    --onefile-no-compression ^
    --output-dir=dist ^
    --output-filename=MisterMobileCropper-v3.exe ^
    --include-package=rembg ^
    --include-package=onnxruntime ^
    --include-package=PIL ^
    --include-package=pillow_heif ^
    --include-package=cv2 ^
    --include-package=numpy ^
    --include-package=pooch ^
    --include-package=numba ^
    --include-package=llvmlite ^
    --nofollow-import-to=scipy ^
    --nofollow-import-to=pytest ^
    --nofollow-import-to=setuptools ^
    --windows-console-mode=force ^
    processor.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Build failed! Check errors above.
    pause
    exit /b 1
)

echo.
echo [POST] Final cleanup...

REM Keep runtime files next to the single exe (as usual workflow)
copy "MM Watermark.png" "dist\MM Watermark.png" >nul 2>&1
copy "INSTRUCTIONS.html" "dist\INSTRUCTIONS.html" >nul 2>&1
if not exist "dist\input" mkdir "dist\input"
if not exist "dist\output" mkdir "dist\output"
if not exist "dist\.u2net" mkdir "dist\.u2net"
if exist ".u2net\u2net.onnx" (
    copy ".u2net\u2net.onnx" "dist\.u2net\u2net.onnx" >nul
) else (
    echo   [WARNING] External model missing: dist\.u2net\u2net.onnx
)

REM Clean build artifacts from dist
if exist "dist\processor.build" rmdir /s /q "dist\processor.build"
if exist "dist\processor.onefile-build" rmdir /s /q "dist\processor.onefile-build"
if exist "dist\processor.dist" rmdir /s /q "dist\processor.dist"
if exist "dist\MisterMobileCropper-v3.dist" rmdir /s /q "dist\MisterMobileCropper-v3.dist"

echo.
echo ============================================
echo   Build complete! v3.0.0
echo ============================================
echo.
echo   dist\
echo     MisterMobileCropper-v3.exe   (single file)
echo     MM Watermark.png
echo     INSTRUCTIONS.html
echo     .u2net\u2net.onnx
echo     input\     (put images here)
echo     output\    (results appear here)
echo.
echo   Users can run directly from dist\ with the usual folder flow.
echo.
pause
