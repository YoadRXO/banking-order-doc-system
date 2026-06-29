#!/usr/bin/env bash
# ============================================================
#  Bank Account AR Detector - launcher for Linux / WSL.
#
#  Usage:
#    ./run.sh                 # demo on sample.png (no webcam needed - best for WSL)
#    ./run.sh --camera 0      # live webcam (needs a working camera + display)
#    ./run.sh --image foo.png # process your own image file
#
#  First run creates the virtualenv and installs deps automatically.
# ============================================================
set -e
cd "$(dirname "$0")"

# First run: create venv + install dependencies.
if [ ! -x ".venv/bin/python" ]; then
    echo "[setup] First run: creating virtual environment..."
    python3 -m venv .venv
    echo "[setup] Installing dependencies (one time, may take a few minutes)..."
    .venv/bin/pip install --upgrade pip >/dev/null
    .venv/bin/pip install -r requirements.txt
fi

if [ "$#" -eq 0 ]; then
    # No arguments -> image demo. Webcams are unreliable under WSL, so this is
    # the path that "just works" and shows the pipeline end to end.
    if [ ! -f sample.png ]; then
        echo "[demo] Generating sample.png..."
        .venv/bin/python tools/make_sample.py
    fi
    echo "[demo] No arguments given -> running the image demo on sample.png."
    echo "       (For the live webcam instead, run:  ./run.sh --camera 0)"
    echo
    exec .venv/bin/python -m account_ar.main --image sample.png
else
    exec .venv/bin/python -m account_ar.main "$@"
fi
