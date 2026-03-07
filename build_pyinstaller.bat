@echo off
setlocal
REM ============================================
REM   PyInstaller Build Script [Fallback]
REM   Compiles processor.py into a stable app folder
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
echo [1/5] Checking PyInstaller...
.\venv\Scripts\python.exe -c "import PyInstaller" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo   Missing PyInstaller. Installing...
    .\venv\Scripts\python.exe -m pip install --no-cache-dir pyinstaller
    if %ERRORLEVEL% NEQ 0 (
        REM Retry import in case pip returned non-zero after partial success.
        .\venv\Scripts\python.exe -c "import PyInstaller" >nul 2>&1
        if %ERRORLEVEL% NEQ 0 (
            echo [ERROR] Failed to install PyInstaller.
            echo         Try manually:
            echo         .\venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
            echo         .\venv\Scripts\python.exe -m pip install --no-cache-dir pyinstaller
            pause
            exit /b 1
        ) else (
            echo   PyInstaller import check passed after pip warning.
        )
    )
) else (
    echo   PyInstaller already installed.
)

echo.
echo [2/5] Cleaning stale PyInstaller artifacts...
if exist "build\pyinstaller" rmdir /s /q "build\pyinstaller"
if exist "dist\pyinstaller" rmdir /s /q "dist\pyinstaller"
if exist "MisterMobileCropper-v3.spec" del /q "MisterMobileCropper-v3.spec"
echo   Cleaned.

echo.
echo [3/5] Preparing external AI model...
if not exist ".u2net" mkdir ".u2net"
if exist "%USERPROFILE%\.u2net\u2net.onnx" (
    copy "%USERPROFILE%\.u2net\u2net.onnx" ".u2net\u2net.onnx" >nul
    echo   Model prepared: .u2net\u2net.onnx
) else (
    echo   [WARNING] u2net.onnx not found at %USERPROFILE%\.u2net\
    echo   The app will download it on first run if internet is available.
)

echo.
echo [4/5] Building with PyInstaller onedir...
.\venv\Scripts\python.exe -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --onedir ^
    --console ^
    --name MisterMobileCropper-v3 ^
    --distpath dist\pyinstaller ^
    --workpath build\pyinstaller ^
    --specpath build\pyinstaller ^
    --collect-all rembg ^
    --collect-all pymatting ^
    --collect-all onnxruntime ^
    --collect-all cv2 ^
    --collect-all pillow_heif ^
    --collect-all PIL ^
    --collect-all numpy ^
    --collect-all pooch ^
    --copy-metadata rembg ^
    --copy-metadata pymatting ^
    --copy-metadata onnxruntime ^
    --copy-metadata pillow_heif ^
    --copy-metadata numpy ^
    --copy-metadata pooch ^
    --hidden-import pymatting ^
    processor.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] PyInstaller build failed! Check errors above.
    pause
    exit /b 1
)

echo.
echo [5/5] Setting up runtime folder...
set "APP_DIR=dist\pyinstaller\MisterMobileCropper-v3"

if not exist "%APP_DIR%" (
    echo [ERROR] Cannot find output app folder: %APP_DIR%
    pause
    exit /b 1
)

copy "MM Watermark.png" "%APP_DIR%\MM Watermark.png" >nul 2>&1
copy "INSTRUCTIONS.html" "%APP_DIR%\INSTRUCTIONS.html" >nul 2>&1
if not exist "%APP_DIR%\input" mkdir "%APP_DIR%\input"
if not exist "%APP_DIR%\output" mkdir "%APP_DIR%\output"
if not exist "%APP_DIR%\.u2net" mkdir "%APP_DIR%\.u2net"
if exist ".u2net\u2net.onnx" (
    copy ".u2net\u2net.onnx" "%APP_DIR%\.u2net\u2net.onnx" >nul
) else (
    echo   [WARNING] External model missing in %APP_DIR%\.u2net\
)

echo.
echo ============================================
echo   Build complete! PyInstaller fallback
echo ============================================
echo.
echo   %APP_DIR%\
echo     MisterMobileCropper-v3.exe
echo     MM Watermark.png
echo     INSTRUCTIONS.html
echo     .u2net\u2net.onnx
echo     input\   put images here
echo     output\  results appear here
echo.
pause
