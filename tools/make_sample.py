"""Generate a synthetic Hebrew banking document for end-to-end testing.

Creates sample.png with several account labels + numbers so you can test the
full OCR -> detect -> order pipeline without a camera or a real document:

    python tools/make_sample.py
    python -m account_ar.main --image sample.png

Only needs Pillow + python-bidi (already in requirements.txt).
"""
from __future__ import annotations

import os
import sys

from PIL import Image, ImageDraw, ImageFont, features

try:
    from bidi.algorithm import get_display
except Exception:  # pragma: no cover
    def get_display(s):
        return s

# Pillow built with libraqm already does BIDI/RTL reordering inside draw.text(),
# so calling get_display() first would reverse Hebrew a *second* time (letters
# come out backwards). Only reorder manually when raqm is unavailable.
_HAS_RAQM = features.check("raqm")


def shape(text: str) -> str:
    """Return `text` ready to draw: logical order if raqm handles BIDI, else visual."""
    return text if _HAS_RAQM else get_display(text)


FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/David.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]

# (Hebrew label, account number). Out of order on purpose so ranking has work to do.
ROWS = [
    ("מס׳ חשבון", "291039"),
    ("חשבון בנק", "292039"),
    ("מס ח.ן", "290134"),
    ("סניף/חשבון", "287512"),
]


def _font(size: int):
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    print("[warn] no Hebrew-capable TTF font found; text may render as boxes.")
    return ImageFont.load_default()


def main() -> int:
    width, height = 900, 650
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    title_font = _font(40)
    label_font = _font(34)
    num_font = _font(40)

    # Title (Hebrew, RTL).
    title = shape("דוגמת מסמך בנקאי")
    draw.text((width - 360, 30), title, font=title_font, fill="black")
    draw.line((40, 100, width - 40, 100), fill="black", width=2)

    # Each row: Hebrew label on the right, the number to its left (LTR digits).
    for i, (label, number) in enumerate(ROWS):
        y = 160 + i * 110
        draw.text((width - 320, y), shape(label), font=label_font, fill="black")
        draw.text((220, y - 2), number, font=num_font, fill="black")

    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sample.png"))
    img.save(out)
    print(f"Wrote {out}")
    print("Now run:  python -m account_ar.main --image sample.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
