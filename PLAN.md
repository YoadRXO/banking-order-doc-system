# Bank Account AR Detector — Project Plan

**Goal:** Point a Windows laptop/webcam at a printed Hebrew banking document. The app
finds every **bank account number** on the page (anchored by Hebrew labels such as
`מס׳ חשבון`, `מס ח.ן`, `חשבון בנק`, `סניף/חשבון`), draws **wireframe AR boxes** around
each one on the live camera feed, and labels them with the **correct sort order**
(ascending account number). Ships as a standalone Windows `.exe`.

> Example: page contains `291039`, `292039`, `290134`.
> Correct order shown on screen → **1) 290134, 2) 291039, 3) 292039**.

> **Multi-document mode:** show *two (or more) separate papers at once* and the app also
> tells you **which paper to put on top of the stack** — the document whose number is first
> in the order goes on top. See §5.

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

## 5. Multi-document stacking order — which paper goes on top

**Use case:** The user shows **two (or more) separate printed documents at once** — for
example two account letters laid side by side under the camera. Each document has its own
bank account number. The app must tell the user **the order to physically stack the papers**:
the document whose number comes **first in the sort order goes on TOP of the stack**, the next
one under it, and so on.

**Worked example (from the request):**

> Right document shows `29200`, left document shows `29201`.
> Ascending order → `29200` is **#1**.
> On screen: the **right** document is tagged **"① TOP OF STACK"**, the **left** is tagged **"②"**.
> Instruction line: **"Put 29200 on top, then 29201 under it."**

Note that the **position on the table (right vs left) does NOT decide the order** — the
**account number** does. In this example the smaller number happens to be on the right, so the
app tells the user to pick up the *right* paper first and place it on top.

**How it works (v1 — reuses the existing pipeline):**

1. OCR + detect account numbers on the frame exactly as today.
2. **Group detections into documents.** Split the accepted numbers into spatial clusters — in
   the common two-paper case this is simply *left group vs right group* (a large horizontal gap
   between them), or by detecting each paper's rectangular edge with OpenCV. Each cluster = one
   physical document, and it takes the account number found inside it.
3. **Order the documents** by their representative account number using the existing
   `ordering.py` (ascending by default; the `ascending` toggle flips which end is "top").
4. **Show the stacking instruction:**
   - Tag each document region on the live frame with its stack position: **"① TOP"**, "②", "③"…
   - Show a side panel listing the stack **top → bottom**, e.g. `TOP → 29200 → 29201 → bottom`.
   - Show one plain-language line: *"Put 29200 on top, then 29201 under it."*

**Edge cases to handle:**

- **More than two documents** — same logic, N clusters, ranks `1..N`, first = top.
- **A document with several numbers** — use the account-labeled number as that document's number.
- **Only one document visible** — fall back to today's single-page behavior (no stack message).
- **Ambiguous / overlapping papers** — if the clusters can't be separated confidently, show a
  hint ("separate the documents / leave a gap") instead of guessing a wrong order.
- **Ascending vs descending** — the `ascending` config flag decides which number ends up on top.

**New/changed code (small):**

- `detector.py` (or a new `grouping.py`): cluster accepted numbers into per-document groups.
- `overlay.py`: draw the per-document "① TOP" tag + the top→bottom stack panel + instruction line.
- `config.json`: e.g. `"stack_order_enabled": true`, `"stack_gap_factor"` (how big a gap splits documents).
- Tests: a case with two groups (right `29200` / left `29201`) → expect **top = 29200**.

---

## 6. Phased delivery

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

### Phase 4.5 — Multi-document stacking order (which paper goes on top) ✅
- **Auto-zoom (ROI) removed** as the default (`roi_enabled=false`) so the whole frame is
  read and several pages are visible at once; still available on the `t` key.
- `page_detect.py`: OpenCV finds each **page** (bright paper rectangle on a darker desk);
  `grouping.py` ties every account number to the page it sits on (numbers on no detected
  page fall back to single-linkage clustering by text-height gap). Degrades gracefully to
  number-only clustering when no page is found.
- `overlay.py`: outlines each page, tags the top one green **"#1 TOP OF STACK"** (orange for
  the rest), a top→bottom stack panel, a plain instruction line, and a **bold green arrow
  ("TAKE THIS FIRST") pointing at the top page** plus a faint 1→2→3 order chain.
- Toggle with `s` live or `--stack` in image mode. Config: `stack_order_enabled`,
  `stack_gap_factor`, `detect_pages`, `page_min_area_frac`, `page_max_area_frac`.
- Tested in `tests/test_grouping.py` (incl. right-`29200`/left-`29201` + page assignment) and
  `tests/test_page_detect.py` (OpenCV, skipped if absent).
- See §5.

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

## 7. Project layout

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

## 8. Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| OCR slow on CPU | Downscale frame; worker thread; process latest frame only; tune `--psm` |
| Hebrew OCR misreads | Anchor on labels not raw text; Phase 5 custom training |
| `.exe` huge / model download at runtime | Bundle models in build (`tools/download_models.py`), one-folder build |
| Numbers that aren't accounts (dates, sums) | Require Hebrew label nearby; length filter; Phase 4 rejection rules |
| Hebrew can't render via OpenCV | Pillow + `python-bidi` (`overlay.py`) |
| Camera differs per machine | `--camera N` flag, list/select devices |
| Multiple documents grouped wrongly | Cluster by clear gaps / paper edges; if ambiguous, ask the user to separate the papers instead of guessing (§5) |

---

## 9. How to run / build (summary)

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
