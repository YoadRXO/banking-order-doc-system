"""Check that the Tesseract engine + Hebrew language data are available.

Tesseract ships its own language data with the installer, so there is nothing to
download at the Python level. This script just verifies the setup (auto-detecting
the Windows install path so it works even if Tesseract isn't on PATH yet):

    python tools/download_models.py
"""
from __future__ import annotations

import os
import sys

# Allow running as `python tools/download_models.py` from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main() -> int:
    try:
        import pytesseract
    except ImportError:
        print("pytesseract is not installed. Run: pip install -r requirements.txt")
        return 1

    # Reuse the app's auto-detection of the Windows install path.
    try:
        from account_ar.ocr_engine import find_tesseract_cmd

        cmd = find_tesseract_cmd()
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
            print(f"Using Tesseract at: {cmd}")
    except Exception:
        pass

    try:
        version = pytesseract.get_tesseract_version()
    except Exception as exc:
        print("Tesseract engine NOT found.")
        print("  Windows: install from https://github.com/UB-Mannheim/tesseract/wiki "
              "(tick 'Hebrew')")
        print("  Linux/WSL: sudo apt install tesseract-ocr tesseract-ocr-heb")
        print("  If it's installed in a custom folder, set 'tesseract_cmd' in config.json.")
        print(f"  Details: {exc}")
        return 1

    langs = pytesseract.get_languages(config="")
    print(f"Tesseract {version} found. Languages: {', '.join(sorted(langs))}")
    if "heb" not in langs:
        print("WARNING: Hebrew ('heb') language data is missing!")
        print("  Windows: re-run the installer and tick 'Hebrew', or drop heb.traineddata")
        print("    into C:\\Program Files\\Tesseract-OCR\\tessdata\\")
        print("  Linux/WSL: sudo apt install tesseract-ocr-heb")
        return 1

    print("Hebrew language data is installed. You're ready to go.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
