"""Draw the AR overlay: wireframe boxes around account numbers, rank tags, and an
ordered side panel. Hebrew text is rendered via Pillow + bidi (OpenCV can't).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from .types import AccountNumber

# Rank colours (BGR) cycled for the boxes/tags.
_PALETTE = [
    (0, 215, 255),   # amber
    (0, 255, 0),     # green
    (255, 128, 0),   # blue
    (255, 0, 255),   # magenta
    (0, 0, 255),     # red
    (255, 255, 0),   # cyan
]


def _color(rank: Optional[int]) -> Tuple[int, int, int]:
    if not rank:
        return (200, 200, 200)
    return _PALETTE[(rank - 1) % len(_PALETTE)]


class _HebrewRenderer:
    """Lazily-loaded Pillow font renderer for RTL Hebrew strings."""

    def __init__(self):
        self._font_cache: dict = {}
        self._bidi = None
        self._init_bidi()

    def _init_bidi(self):
        # Pillow built with libraqm already reorders RTL/BIDI inside draw.text(),
        # so reordering here too would reverse Hebrew a second time (backwards
        # letters). Only fall back to manual visual ordering when raqm is absent.
        from PIL import features

        if features.check("raqm"):
            self._bidi = lambda s: s
            return
        try:
            from bidi.algorithm import get_display
            self._bidi = get_display
        except Exception:
            self._bidi = lambda s: s  # fall back to raw order

    def _font(self, size: int):
        from PIL import ImageFont

        if size in self._font_cache:
            return self._font_cache[size]
        font = None
        for path in (
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/David.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ):
            try:
                font = ImageFont.truetype(path, size)
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()
        self._font_cache[size] = font
        return font

    def draw(self, frame, text: str, org: Tuple[int, int], size: int, color_bgr):
        """Draw `text` (may be Hebrew) at pixel `org` (top-left). Returns frame."""
        import numpy as np
        from PIL import Image, ImageDraw

        img = Image.fromarray(frame[:, :, ::-1])  # BGR→RGB
        draw = ImageDraw.Draw(img)
        shaped = self._bidi(text)
        draw.text(org, shaped, font=self._font(size), fill=tuple(int(c) for c in color_bgr[::-1]))
        frame[:, :] = np.asarray(img)[:, :, ::-1]  # RGB→BGR
        return frame


class Overlay:
    def __init__(self):
        self._hebrew = _HebrewRenderer()

    def draw(
        self,
        frame,
        accounts: List[AccountNumber],
        status: str = "",
        scale: float = 1.0,
        sharpness: Optional[float] = None,
        good_sharpness: float = 75.0,
        min_sharpness: float = 30.0,
    ):
        """Render boxes + ranks for `accounts` onto `frame` in place.

        `scale` maps OCR-frame coordinates to the current display frame.
        When `sharpness` is given, a live focus meter is drawn so the user can see
        whether the frame is sharp enough to read.
        """
        import cv2

        if sharpness is not None:
            self._draw_focus(frame, sharpness, good_sharpness, min_sharpness)

        for acc in sorted(accounts, key=lambda a: (a.rank or 999)):
            color = _color(acc.rank)
            pts = [(int(x * scale), int(y * scale)) for (x, y) in acc.detection.polygon]

            # Wireframe box.
            import numpy as np
            poly = np.array(pts, dtype="int32")
            cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=3)

            x1 = min(p[0] for p in pts)
            y1 = min(p[1] for p in pts)

            # Rank tag (e.g. "#1  290134") above the box.
            tag = f"#{acc.rank}  {acc.digits}" if acc.rank else acc.digits
            self._draw_tag(frame, tag, (x1, y1 - 30), color)

        self._draw_panel(frame, accounts)
        if status:
            self._draw_status(frame, status)
        return frame

    def _draw_tag(self, frame, text: str, org, color):
        import cv2

        x, y = org
        y = max(y, 24)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x, y - th - 8), (x + tw + 10, y + 6), color, -1)
        cv2.putText(frame, text, (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    def _draw_panel(self, frame, accounts: List[AccountNumber]):
        import cv2

        h, w = frame.shape[:2]
        ranked = sorted([a for a in accounts if a.rank], key=lambda a: a.rank)
        lines = ["ORDER (asc):"] + [f"{a.rank}. {a.digits}" for a in ranked]
        if not ranked:
            lines = ["ORDER (asc):", "(no accounts detected)"]

        panel_w = 230
        panel_h = 24 * len(lines) + 16
        x0 = w - panel_w - 12
        y0 = 12
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        y = y0 + 26
        for i, line in enumerate(lines):
            color = (255, 255, 255) if i == 0 else _color(i)
            cv2.putText(frame, line, (x0 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            y += 24

    def _draw_status(self, frame, status: str):
        import cv2

        h = frame.shape[0]
        cv2.putText(frame, status, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

    def _draw_focus(self, frame, sharpness: float, good: float, minv: float):
        """Top-left focus meter: a bar that fills toward green as the frame gets sharper.

        Red + 'BLURRY' below the OCR threshold, orange while soft, green when sharp.
        Gives the user immediate feedback to fix focus/distance instead of guessing.
        """
        import cv2

        if sharpness >= good:
            color, label = (0, 220, 0), "FOCUS OK"
        elif sharpness >= minv:
            color, label = (0, 165, 255), "FOCUS: soft - hold steady"
        else:
            color, label = (0, 0, 255), "BLURRY - move back / tap to focus"

        x0, y0, bw, bh = 12, 14, 260, 20
        frac = max(0.05, min(1.0, sharpness / float(good)))
        cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), (60, 60, 60), -1)
        cv2.rectangle(frame, (x0, y0), (x0 + int(bw * frac), y0 + bh), color, -1)
        cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), (220, 220, 220), 1)
        cv2.putText(frame, label, (x0, y0 + bh + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
