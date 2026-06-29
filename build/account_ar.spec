# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the Bank Account AR Detector (Tesseract OCR).
# Build (on Windows, from the repo root):
#     build\build_exe.bat                       # or: pyinstaller build/account_ar.spec
#     -> dist/account_ar/account_ar.exe
#
# One-FOLDER build. Lightweight (no torch). The user must have the Tesseract engine
# installed on the machine that RUNS the .exe (UB-Mannheim installer, with Hebrew).
# Optionally bundle a `tessdata` folder beside the build to ship Hebrew data with it.

import os
from PyInstaller.utils.hooks import collect_data_files

ROOT = os.path.abspath(os.getcwd())

datas = []
binaries = []
hiddenimports = ["account_ar"]

datas += collect_data_files("bidi")

# Optionally ship Hebrew traineddata: copy tesseract's tessdata into ./tessdata
# before building, and it will be bundled and used automatically.
tessdata_dir = os.path.join(ROOT, "tessdata")
if os.path.isdir(tessdata_dir):
    datas.append((tessdata_dir, "tessdata"))

if os.path.isfile(os.path.join(ROOT, "config.json")):
    datas.append((os.path.join(ROOT, "config.json"), "."))

# The accepted-account-labels list (editable beside the .exe; bundled as a default).
if os.path.isfile(os.path.join(ROOT, "accepted_labels.txt")):
    datas.append((os.path.join(ROOT, "accepted_labels.txt"), "."))


block_cipher = None

a = Analysis(
    [os.path.join(ROOT, "account_ar", "main.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="account_ar",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="account_ar",
)
