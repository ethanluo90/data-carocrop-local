@echo off
setlocal
REM ============================================
REM   Nuitka Build Script [Stable]
REM   Compiles processor.py into a standalone app folder
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
    echo   The app will download it on first run if internet is available.
)

echo.
echo [4/4] Building standalone executable...
.\venv\Scripts\python.exe -m nuitka ^
    --standalone ^
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
echo [POST] Setting up runtime folder...
set "DIST_APP=dist\MisterMobileCropper-v3.dist"
if not exist "%DIST_APP%" set "DIST_APP=dist\processor.dist"

if not exist "%DIST_APP%" (
    echo [ERROR] Cannot find output app folder in dist.
    pause
    exit /b 1
)

copy "MM Watermark.png" "%DIST_APP%\MM Watermark.png" >nul 2>&1
copy "INSTRUCTIONS.html" "%DIST_APP%\INSTRUCTIONS.html" >nul 2>&1
if not exist "%DIST_APP%\input" mkdir "%DIST_APP%\input"
if not exist "%DIST_APP%\output" mkdir "%DIST_APP%\output"
if not exist "%DIST_APP%\.u2net" mkdir "%DIST_APP%\.u2net"
if exist ".u2net\u2net.onnx" (
    copy ".u2net\u2net.onnx" "%DIST_APP%\.u2net\u2net.onnx" >nul
) else (
    echo   [WARNING] External model missing in %DIST_APP%\.u2net\
)

if exist "dist\processor.build" rmdir /s /q "dist\processor.build"
if exist "dist\processor.onefile-build" rmdir /s /q "dist\processor.onefile-build"

echo.
echo ============================================
echo   Build complete! v3.0.0 [Stable]
echo ============================================
echo.
echo   %DIST_APP%\
echo     MisterMobileCropper-v3.exe
echo     MM Watermark.png
echo     INSTRUCTIONS.html
echo     .u2net\u2net.onnx
echo     input\   put images here
echo     output\  results appear here
echo.
echo   For onefile testing, use build_onefile_experimental.bat
echo.
pause

