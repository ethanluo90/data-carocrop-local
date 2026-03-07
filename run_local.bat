@echo off
echo ============================================
echo   MisterMobile - Carousell Image Cropper
echo   Format: 1080x1080px Square
echo ============================================
echo.

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

echo Press any key to start processing...
pause >nul
echo.
echo Running cropper...
echo.

.\venv\Scripts\python processor.py

echo.
echo ============================================
echo   Processing Complete!
echo ============================================
echo.
echo Output folder: output\
echo.
pause
