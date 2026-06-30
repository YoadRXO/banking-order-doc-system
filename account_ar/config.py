"""Runtime settings, Hebrew keyword lists, and resource-path helpers.

Stdlib-only so it can be imported by the pure-logic modules and their tests.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Optional


# --- Hebrew account-number labels we anchor on -----------------------------
# Raw forms (as printed). text_utils normalizes both sides before matching, so
# punctuation / geresh / spacing variants are handled automatically.
# A document line counts as an account line when the DISTINCTIVE word of one of these
# phrases — the account noun "חשבון" (the longest word) — appears on it. The short
# "מספר"/"מס" is NOT required (OCR often drops it on live video), but "חשבון" is
# matched strictly enough to exclude savings ("חסכון") and branch ("סניף") lines.
DEFAULT_LABEL_KEYWORDS: List[str] = [
    "מספר חשבון",
    "מספר חשבון ראשי",
    "מס חשבון",
    "מס חשבון ראשי",
    "מס׳ חשבון",
    "מס' חשבון",
    "סניף/חשבון",
]

# The account noun used to locate the label on a line (so the number after it — its
# value side — can be taken).
DEFAULT_ACCOUNT_WORD: str = "חשבון"

# "Number" words — only consulted when require_number_word is on (extra-strict mode).
DEFAULT_NUMBER_WORDS: List[str] = ["מספר", "מס"]

# Box-level anchors, used only by the is_label() debug helper (not by detection).
DEFAULT_ANCHOR_TOKENS: List[str] = [
    "חשבון",  # "account"
    "חן",     # "ח.ן" / "ח-ן" abbreviation after normalization
]


@dataclass
class Settings:
    # OCR (Tesseract). Tesseract language codes: Hebrew = "heb", English = "eng".
    languages: List[str] = field(default_factory=lambda: ["heb", "eng"])
    tesseract_cmd: Optional[str] = None   # full path to tesseract.exe (None = autodetect/PATH)
    tessdata_dir: Optional[str] = None    # folder with heb.traineddata (None = default)
    tesseract_oem: int = 3                # OCR engine mode (3 = default LSTM)
    tesseract_psm: int = 6                # page segmentation mode (6 = uniform block: keeps label+number on one row)
    ocr_max_width: int = 2000             # downscale wider frames before OCR (speed)
    ocr_min_width: int = 2000             # upscale narrower frames before OCR (accuracy on small/low-res text)
    preprocess: str = "clahe"             # none|gray|clahe|otsu|adaptive — cleanup before OCR
    min_ocr_confidence: float = 0.20

    # Focus gating (live mode). A blurry/out-of-focus frame has no readable text in it,
    # so OCRing it just wastes ~1-2s and produces noise. We measure sharpness (variance
    # of the Laplacian) on each frame, normalized to a fixed width so the threshold is
    # the same for any camera resolution. Below `min_sharpness` we SKIP OCR and tell the
    # user to adjust focus/distance; at/above `good_sharpness` the on-screen meter is green.
    sharpness_norm_width: int = 1000      # resize width before measuring sharpness
    min_sharpness: float = 30.0           # skip OCR below this (frame too blurry to read)
    good_sharpness: float = 75.0          # focus meter turns green at/above this

    # Account-number shape
    min_digits: int = 4
    max_digits: int = 13
    exact_lengths: List[int] = field(default_factory=list)  # e.g. [6,9]; empty = any in range

    # Label matching
    label_keywords: List[str] = field(default_factory=lambda: list(DEFAULT_LABEL_KEYWORDS))
    anchor_tokens: List[str] = field(default_factory=lambda: list(DEFAULT_ANCHOR_TOKENS))
    account_word: str = DEFAULT_ACCOUNT_WORD   # the noun a label must contain ("חשבון")
    number_words: List[str] = field(default_factory=lambda: list(DEFAULT_NUMBER_WORDS))
    require_number_word: bool = False  # extra-strict: also demand "מספר"/"מס" on the line.
                                       # Default off — more robust when OCR drops that word.
    label_fuzzy_threshold: float = 0.82
    account_word_threshold: float = 0.70   # how close an OCR word must be to "חשבון" to count.
                                           # Lower = catches more garbled live-OCR reads, while
                                           # still excluding "חסכון" (savings, ~0.60 similar).

    # Spatial association (all scaled by label height → resolution independent)
    line_overlap_ratio: float = 0.30   # min vertical overlap to count as "same row"
    max_horizontal_gap_factor: float = 9.0   # × label height
    below_gap_factor: float = 2.2            # × label height
    value_side: str = "left"           # which side of a "…חשבון" label the account number sits on:
                                       # "left" (Hebrew RTL — the number comes after the label, to its
                                       # left), "right" (LTR), or "any" to accept either side. Numbers on
                                       # the wrong side of the label are ignored.

    # Modes
    accept_unlabeled: bool = False     # accept numbers with no nearby label
    ascending: bool = True             # sort order
    track_seconds: float = 3.0         # live mode: keep an account on screen this long after
                                       # its last sighting, so one good frame "sticks" even when
                                       # most frames (glare/blur/angle) are unreadable. 0 = off.
    confirm_sightings: int = 2         # SAFETY: only show a number after it has been read this
                                       # many times (same value) within track_seconds. Stops a
                                       # one-off misread (glare/rotation) from flashing a wrong
                                       # account. 1 = show immediately (less safe, more "instant").
    lock_found: bool = False           # accumulate mode: freeze each found number in place and
                                       # never expire it, so they pile up as you pan across papers
                                       # (instead of refreshing away). Toggle with 'l'; 'c' clears.

    # Magnifier target box ("ROI"). Instead of OCRing the whole frame, read only a
    # central box (drawn on screen) and digitally zoom into it. Lets the user aim the
    # account-number line into the box from a bit further back, and avoids picking up
    # other numbers elsewhere on the page. Press 't' to toggle, 'r' is rotation.
    roi_enabled: bool = True
    roi_width_frac: float = 0.70       # box width as a fraction of the frame
    roi_height_frac: float = 0.24      # box height (a horizontal band suits one text line)

    # Camera / UI
    camera_index: int = 0
    camera_width: int = 1280              # request a higher-res capture for readable text
    camera_height: int = 720
    camera_rotation: int = 0              # rotate every frame by 0/90/180/270 deg so the page
                                          # reads upright (some phone-as-webcam feeds come in
                                          # flipped). Press 'r' live to cycle until it looks right.
    window_name: str = "Bank Account AR Detector"

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Settings":
        """Load defaults, overlay ./config.json, then overlay accepted_labels.txt.

        `accepted_labels.txt` (if present) is the single source of truth for which
        words accept a number as a bank account — it overrides label_keywords /
        anchor_tokens from config.json.
        """
        settings = cls()
        candidate = path or os.path.join(os.getcwd(), "config.json")
        if os.path.isfile(candidate):
            with open(candidate, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for key, value in data.items():
                if hasattr(settings, key):
                    setattr(settings, key, value)

        accepted = load_accepted_labels()
        if accepted:
            settings.label_keywords = accepted
            # A single-word entry is a strong anchor (accepts the number on its line).
            settings.anchor_tokens = [w for w in accepted if " " not in w]
        return settings

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


ACCEPTED_LABELS_FILE = "accepted_labels.txt"


def load_accepted_labels(path: Optional[str] = None) -> List[str]:
    """Read the accepted account-label words from `accepted_labels.txt`.

    One label per line; `#` comments and blank lines are ignored. Looked up next to
    the working dir first (matching config.json), then alongside the bundled app so
    a packaged .exe still finds its defaults. Returns [] if no file exists.
    """
    candidates = []
    if path:
        candidates.append(path)
    else:
        candidates.append(os.path.join(os.getcwd(), ACCEPTED_LABELS_FILE))
        candidates.append(resource_path(ACCEPTED_LABELS_FILE))

    for candidate in candidates:
        if not os.path.isfile(candidate):
            continue
        labels: List[str] = []
        with open(candidate, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                labels.append(line)
        return labels
    return []


def resource_path(relative: str) -> str:
    """Resolve a bundled resource path, working both in dev and in a PyInstaller .exe."""
    base = getattr(sys, "_MEIPASS", os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    return os.path.join(base, relative)


def default_tessdata_dir() -> Optional[str]:
    """When frozen, point Tesseract at a bundled `tessdata/` dir if present; else None."""
    if getattr(sys, "frozen", False):
        bundled = resource_path("tessdata")
        if os.path.isdir(bundled):
            return bundled
    return None
