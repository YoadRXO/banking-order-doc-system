# Bank Account AR Detector

Point your webcam at a printed **Hebrew** banking document. The app finds every
**bank account number** (anchored by Hebrew labels like `מס׳ חשבון`, `מס ח.ן`,
`חשבון בנק`, `סניף/חשבון`), draws wireframe **AR boxes** around each one on the live
video, and labels them with the **correct ascending order**.

> `291039`, `292039`, `290134`  →  **#1 290134 · #2 291039 · #3 292039**

**OCR engine:** [Tesseract](https://github.com/tesseract-ocr/tesseract) with the
`heb` language pack — CPU-only, offline, and packages cleanly into a Windows `.exe`.
(EasyOCR has no Hebrew model; modern Surya needs a Docker/GPU server — neither fits a
standalone desktop app. GPU is revisited in Phase 5 for training.)

See **[PLAN.md](PLAN.md)** for the full phased plan and architecture.

---

## Prerequisite: install the Tesseract engine (with Hebrew)

`pytesseract` is only a wrapper — you need the actual Tesseract program + Hebrew data:

- **Windows:** [UB-Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki)
  → during setup, expand "Additional language data" and tick **Hebrew**.
- **Linux / WSL:** `sudo apt install tesseract-ocr tesseract-ocr-heb`

Verify: `python tools/download_models.py` (prints the version and confirms `heb`).

## Install (Python side)

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate     Linux/Mac:  source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python -m account_ar.main             # live camera
python -m account_ar.main --camera 1  # pick a different webcam
```

**Keys:** `q`/`ESC` quit · `space` screenshot · `p` pause OCR · `a` toggle
accept-unlabeled-numbers · `b` cycle preprocessing (`clahe`→`none`→`gray`→`otsu`→`adaptive`)
· `c` clear remembered accounts.

### Tips for low-quality / webcam capture

Cheap webcams photographing faint Hebrew bank print need contrast help, so the
default preprocessing is **`clahe`** (local contrast equalization) and frames are
upscaled to ~2000 px before OCR. Because most hand-held frames are unreadable
(glare, blur, motion) the app **remembers** each account for `track_seconds` (2.5 s)
after its last sighting — so a single good frame locks the result on screen. Hold
the page flat, fill the frame with it, and avoid glare; press `c` when you switch
documents. Only **account-number** lines are tagged (those reading `מספר חשבון` /
`מס חשבון` / `מספר חשבון ראשי`) — branch/savings/date/amount numbers are ignored.

### Test on an image (no webcam needed)

```bash
python tools/make_sample.py                      # creates sample.png
python -m account_ar.main --image sample.png     # prints accounts + saves sample_annotated.png
```

## Tests (pure logic — no webcam, no OCR engine)

```bash
python -m unittest discover -s tests -v
```

## Build the Windows `.exe`

Run **on Windows** from the repo root:

```bat
build\build_exe.bat
```

Makes a venv, installs deps, checks Tesseract, and runs PyInstaller →
`dist\account_ar\account_ar.exe`. Ship the whole **`dist\account_ar` folder**.
The machine that runs the `.exe` also needs the **Tesseract engine** installed with
Hebrew (or bundle a `tessdata` folder — see `build/account_ar.spec`).

## How detection works

1. **Preprocess** — grayscale + **CLAHE** contrast boost + upscale, so faint print
   on cheap cameras survives OCR (the single biggest accuracy win here).
2. **OCR** (Tesseract `heb`+`eng`, PSM 6) reads all words with bounding boxes.
3. **Lines** — the word boxes are grouped into text lines by vertical overlap.
4. **Label match (line-level)** — a line is an account line when the account noun
   `חשבון` appears on it (matched with OCR-noise tolerance, but strictly enough to
   exclude savings `חסכון` and branch `סניף`). The short `מספר`/`מס` is **not**
   required — live OCR often drops it — so `מספר חשבון`, `מס חשבון ראשי`, and even a
   bare/garbled `חשבון` all match. (Set `require_number_word: true` in `config.json`
   to also demand the `מספר`/`מס` word.)
5. **Number on the value side** — on an account line the account number is the one
   *after* the label, i.e. to its **left** in Hebrew RTL. Numbers on the wrong side,
   and every non-numeric word/character on the line, are ignored. (Set `value_side`
   to `right` or `any` in `config.json` for other layouts.)
6. **Tracking** (live mode) remembers each account for `track_seconds` so one good
   frame sticks across the unreadable ones.
7. **Ordering** sorts numerically ascending and assigns ranks `1..n`.
8. **Overlay** draws boxes + ranks + an ordered side panel.

### Which words accept a number — `accepted_labels.txt`

The accepted account labels live in one plain-text file at the repo root:
**`accepted_labels.txt`** (one **full phrase** per line; `#` lines are comments).
A line of the document is treated as an account line **only** when every word of one
of these phrases appears on it (the words may be split across OCR boxes) — e.g.
`מספר חשבון`, `מס חשבון`, `מספר חשבון ראשי`. A bare `חשבון`, or branch/savings lines,
do **not** qualify. The number *after* the label on that line is taken; everything
else is ignored. This file is the single source of truth — if present it overrides
`label_keywords` in `config.json`. Add or remove a phrase, save, rerun — no code
changes. Delete the file to fall back to `config.json`.

Edit **`config.json`** for the rest: expected digit lengths (`exact_lengths`, e.g.
`[6, 9]`), preprocessing, `track_seconds`, or point `tesseract_cmd` at a custom
install path.

## Roadmap → training on real documents

Phase 5 (see PLAN.md) fine-tunes a **YOLOv8** field detector on photographed real
forms to locate the account-number region directly — and that's where **GPU** has a
real home. **Never commit real account numbers** — keep datasets local (already in
`.gitignore`).
