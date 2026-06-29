@echo off
REM ============================================================
REM  Bank Account AR Detector - one-click launcher (Windows)
REM
REM  Just double-click this file. The first run sets up Python
REM  dependencies automatically; later runs start instantly.
REM
REM  You can also pass options, e.g.:
REM    run.bat --camera 1            (use a different webcam)
REM    run.bat --image sample.png   (process an image, no webcam)
REM ============================================================
setlocal
cd /d "%~dp0"

set "FRESH="
if not exist ".venv\Scripts\python.exe" (
    echo [setup] First run: creating virtual environment...
    python -m venv .venv || goto :err
    set "FRESH=1"
)

call .venv\Scripts\activate

if defined FRESH (
    echo [setup] Installing dependencies ^(one time, may take a few minutes^)...
    python -m pip install --upgrade pip
    pip install -r requirements.txt || goto :err
)

REM Make the Tesseract engine findable (it is usually not on PATH after install).
if exist "C:\Program Files\Tesseract-OCR\tesseract.exe" set "PATH=C:\Program Files\Tesseract-OCR;%PATH%"

echo.
echo [run] Starting Bank Account AR Detector...
echo       Keys:  q/ESC quit  ^|  space screenshot  ^|  p pause  ^|  a accept-unlabeled
echo.
python -m account_ar.main %*
goto :end

:err
echo.
echo [error] Setup failed. Make sure Python 3 is installed and on your PATH.
echo         You also need the Tesseract engine installed with Hebrew:
echo         https://github.com/UB-Mannheim/tesseract/wiki
pause
exit /b 1

:end
endlocal
pause
