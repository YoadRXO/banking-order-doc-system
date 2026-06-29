# Bank Account AR Detector — Project Plan

**Goal:** Point a Windows laptop/webcam at a printed Hebrew banking document. The app
finds every **bank account number** on the page (anchored by Hebrew labels such as
`מס׳ חשבון`, `מס ח.ן`, `חשבון בנק`, `סניף/חשבון`), draws **wireframe AR boxes** around
each one on the live camera feed, and labels them with the **correct sort order**
(ascending account number). Ships as a standalone Windows `.exe`.

> Example: page contains `291039`, `292039`, `290134`.
> Correct order shown on screen → **1) 290134, 2) 291039, 3) 292039**.

---

## 1. What "AR" means here

This is **camera-overlay AR** (a.k.a. video see-through), not 3D/marker AR. We:

1. Grab live frames from the webcam (OpenCV).
2. Run OCR + field detection on the frame.
3. Draw boxes + rank labels back onto the live frame, registered to where the
   numbers actually are.

No ARKit/ARCore/markers needed. This is the right scope for "show a paper to the
camera and see boxes + order".

---

## 2. Architecture

```
                 ┌──────────────┐   frames    ┌─────────────────────┐
   Webcam ──────▶│  camera.py   │────────────▶│   pipeline.py       │
                 │ (threaded)   │             │  (OCR worker thread) │
                 └──────────────┘             └─────────┬───────────┘
                                                        │ detections
                                                        ▼
                                              ┌─────────────────────┐
                                              │  ocr_engine.py       │  Tesseract (heb+eng)
                                              └─────────┬───────────┘
                                                        ▼
                       ┌────────────────────────────────────────────┐
                       │ text_utils.py  → normalize Hebrew, find      │
                       │   labels (keywords) + number candidates      │
                       │ detector.py    → associate numbers⇄labels    │
                       │   spatially → list of account numbers        │
                       │ ordering.py    → numeric sort + rank         │
                       └───────────────────────┬──────────────────────┘
                                                ▼
                 ┌──────────────┐  ranked     ┌─────────────────────┐
   Screen ◀──────│  overlay.py  │◀────────────│   main.py (UI loop) │
                 │ boxes+ranks  │             └─────────────────────┘
                 └──────────────┘
```

**Why a worker thread:** OCR takes ~0.2–1 s/frame on CPU. The UI must stay smooth,
so OCR runs on a background thread on the *latest* frame; the UI draws the *most
recent* results every frame at full FPS.

---

## 3. Tech stack

| Concern              | Choice                          | Why |
|----------------------|----------------------------------|-----|
| Camera + drawing     | OpenCV (`opencv-python`)         | Standard, fast, cross-platform |
| OCR (Hebrew)         | **Tesseract** (`heb`,`eng`)      | Real Hebrew support, CPU, offline, packages into a `.exe`. (EasyOCR has NO Hebrew model; modern Surya needs a Docker/GPU server — won't package as a desktop `.exe`.) |
| Hebrew text rendering| Pillow + `python-bidi`           | `cv2.putText` can't render Hebrew/RTL |
| Numerics             | numpy                            | array math |
| Packaging            | **PyInstaller** (one-folder)     | Bundles Python + models into a Windows `.exe` |
| Tests                | `unittest` (stdlib)              | Pure-logic tests, no heavy deps |

**Digits note:** account *numbers* are Latin digits (0–9) even on Hebrew pages.
"Detect only Hebrew" is satisfied by **anchoring on Hebrew labels** — we only accept
a number as an account number when it sits next to a Hebrew account label (or, in a
looser mode, when it matches the expected digit length).

---

## 4. Detection logic (the core)

1. **OCR** the frame → list of `(polygon, text, confidence)`.
2. **Classify** each text box:
   - *Label?* Normalize Hebrew (strip niqqud/geresh/punctuation), then match against
     keyword list (`חשבון`, `מס׳ חשבון`, `ח.ן`, `סניף/חשבון`, …) by anchor-word +
     fuzzy ratio.
   - *Number candidate?* Regex for digit runs (with `- / .` separators), strip to
     digits, keep length in `[min_digits, max_digits]`.
3. **Associate** numbers ⇄ labels spatially (resolution-independent, scaled by label
   height):
   - same row (vertical overlap ≥ 30 %) and horizontally close, **or**
   - directly below the label within a small gap.
   - A box containing *both* a keyword and a number (e.g. `חשבון 292039`) → accept
     its number directly.
4. **Order:** strip to digits → integer ascending sort → assign rank `1..n`.
5. **Render:** box each accepted number, tag with its rank, show an ordered side list.

Optional **"accept unlabeled" mode** (`a` key): accept any number of the expected
length even without a nearby label — useful for messy scans, more false positives.

---

## 5. Phased delivery

### Phase 0 — Scaffolding ✅ (this commit)
- Repo layout, `requirements.txt`, config, `.gitignore`, README, PLAN.

### Phase 1 — Pure logic + tests ✅ (this commit)
- `text_utils` (Hebrew normalize, label match, number extraction)
- `detector` (spatial association)
- `ordering` (numeric rank)
- `unittest` suite covering the worked example. **Runs with zero heavy deps.**

### Phase 2 — OCR + camera + overlay + UI ✅ (this commit)
- `ocr_engine` (Tesseract wrapper via pytesseract, downscale for speed)
- `camera` (threaded capture), `overlay` (boxes, ranks, Hebrew via Pillow)
- `pipeline` (OCR worker thread), `main` (window, keybindings)
- Runnable: `python -m account_ar.main`

### Phase 3 — Windows `.exe` ✅ (this commit, build on Windows)
- `build/account_ar.spec`, `build/build_exe.bat`, `tools/download_models.py`
- One-folder PyInstaller bundle (lightweight, no torch); optional bundled `tessdata`.

### Phase 4 — Accuracy hardening (next)
- Real-world tuning: keyword list expansion, separator handling, ROI/crop, perspective
  de-skew (`cv2.getPerspectiveTransform`), temporal smoothing of boxes across frames.
- Confidence thresholds + reject obvious non-accounts (dates, phone, amounts).

### Phase 5 — Train on real banking documents (your stretch goal)
- **Data:** collect & photograph real forms (varied lighting/angles/banks).
- **Annotate:** label the *account-number field* boxes (and label keywords) with
  Roboflow / LabelImg / Label Studio. Target a few hundred → few thousand boxes.
- **Field detector:** fine-tune **YOLOv8** to locate the account-number region
  directly (more robust than keyword spatial heuristics).
- **OCR fine-tune (optional):** fine-tune Tesseract (or a GPU model) on Hebrew
  banking fonts if generic OCR misreads digits. **GPU has a real home here** —
  training/inference on a server (Docker fine) rather than in the desktop `.exe`.
- **Integration:** YOLO finds the field → crop → OCR reads digits → existing
  ordering/overlay pipeline unchanged.
- **Privacy:** account numbers are sensitive. Keep the dataset local/encrypted, never
  commit real account data, redact in any shared samples.

### Phase 6 — Packaging & polish
- Installer (Inno Setup) wrapping the one-folder build, app icon, splash, settings UI,
  GPU/CPU auto-detect, logging.

---

## 6. Project layout

```
banking-order-doc-system/
├── PLAN.md                  ← this file
├── README.md
├── requirements.txt
├── config.json              ← editable keywords / thresholds (optional override)
├── account_ar/              ← the package
│   ├── config.py            ← Settings + resource paths
│   ├── types.py             ← Detection / AccountNumber dataclasses
│   ├── text_utils.py        ← Hebrew normalize, label match, number extraction
│   ├── detector.py          ← spatial association  (pure logic)
│   ├── ordering.py          ← numeric sort + rank   (pure logic)
│   ├── ocr_engine.py        ← Tesseract OCR wrapper
│   ├── camera.py            ← threaded webcam capture
│   ├── overlay.py           ← AR drawing (boxes, ranks, Hebrew text)
│   ├── pipeline.py          ← OCR worker thread orchestration
│   └── main.py              ← UI loop / entry point
├── tools/
│   └── download_models.py   ← verify Tesseract + Hebrew data are installed
├── build/
│   ├── account_ar.spec      ← PyInstaller spec
│   └── build_exe.bat        ← one-click Windows build
└── tests/
    ├── test_text_utils.py
    ├── test_detector.py
    └── test_ordering.py
```

---

## 7. Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| OCR slow on CPU | Downscale frame; worker thread; process latest frame only; tune `--psm` |
| Hebrew OCR misreads | Anchor on labels not raw text; Phase 5 custom training |
| `.exe` huge / model download at runtime | Bundle models in build (`tools/download_models.py`), one-folder build |
| Numbers that aren't accounts (dates, sums) | Require Hebrew label nearby; length filter; Phase 4 rejection rules |
| Hebrew can't render via OpenCV | Pillow + `python-bidi` (`overlay.py`) |
| Camera differs per machine | `--camera N` flag, list/select devices |

---

## 8. How to run / build (summary)

```bash
# dev (any OS with a webcam)
pip install -r requirements.txt
python -m account_ar.main                 # q=quit, space=snapshot, a=accept-unlabeled

# logic tests (no webcam / no heavy deps needed)
python -m unittest discover -s tests -v

# build Windows .exe (run ON Windows)
python tools/download_models.py           # verify Tesseract + Hebrew are installed
build\build_exe.bat                        # → dist/account_ar/account_ar.exe
```
