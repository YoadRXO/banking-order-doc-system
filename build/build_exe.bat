@echo off
REM One-click Windows build for the Bank Account AR Detector (Tesseract OCR).
REM Run from the repo root:  build\build_exe.bat
REM
REM Lightweight build (no torch). REQUIREMENT: the Tesseract engine must be
REM installed on this machine AND on any machine that runs the .exe:
REM   https://github.com/UB-Mannheim/tesseract/wiki  (tick "Hebrew" during install)

setlocal
cd /d "%~dp0\.."

echo === [1/4] Creating / using virtual env ===
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
)
call .venv\Scripts\activate

echo === [2/4] Installing dependencies ===
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

echo === [3/4] Checking Tesseract + Hebrew ===
python tools\download_models.py

echo === [4/4] Building the .exe with PyInstaller ===
pyinstaller --noconfirm build\account_ar.spec

echo.
echo Build complete:  dist\account_ar\account_ar.exe
echo Ship the whole  dist\account_ar  folder.
echo (Target machines also need the Tesseract engine installed with Hebrew.)
endlocal
pause
