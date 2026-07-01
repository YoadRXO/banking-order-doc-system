"""Draw the AR overlay: wireframe boxes around account numbers, rank tags, and an
ordered side panel. Hebrew text is rendered via Pillow + bidi (OpenCV can't).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from .grouping import stack_instruction
from .types import AccountNumber, DocumentGroup

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
        groups: Optional[List[DocumentGroup]] = None,
    ):
        """Render boxes + ranks for `accounts` onto `frame` in place.

        `scale` maps OCR-frame coordinates to the current display frame.
        When `sharpness` is given, a live focus meter is drawn so the user can see
        whether the frame is sharp enough to read.
        When `groups` is given (multi-document mode), each document is tagged with its
        stacking position and a "which paper goes on top" panel is drawn.
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

        # Multi-document mode: tag each document with its stacking position and show a
        # "which paper goes on top" panel instead of the plain ORDER list.
        if groups:
            self._draw_stack(frame, groups, scale)
            self._draw_direction(frame, groups, scale)
            self._draw_stack_panel(frame, groups)
        else:
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
        # Collapse duplicates to one panel line with a count, but keep their boxes.
        counts: dict = {}
        for a in ranked:
            key = (a.rank, a.digits)
            counts[key] = counts.get(key, 0) + 1
        lines = ["ORDER (asc):"]
        for (rank, digits), n in sorted(counts.items()):
            lines.append(f"{rank}. {digits}" + (f"  x{n}" if n > 1 else ""))
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

    def _draw_stack(self, frame, groups: List[DocumentGroup], scale: float):
        """Outline each page, tag its account with the stacking position; #1 = TOP (green)."""
        import cv2
        import numpy as np

        for doc in groups:
            pos = doc.stack_position or 0
            top = pos == 1
            color = (0, 220, 0) if top else (0, 165, 255)  # top = green, rest = orange

            # Outline the detected page (if any) so the user sees which sheet is which.
            if doc.page_quad:
                page = np.array([(int(x * scale), int(y * scale)) for (x, y) in doc.page_quad],
                                dtype="int32")
                cv2.polylines(frame, [page], isClosed=True, color=color,
                              thickness=6 if top else 3)

            pts = [(int(x * scale), int(y * scale)) for (x, y) in doc.primary.detection.polygon]
            poly = np.array(pts, dtype="int32")
            cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=4)
            x1 = min(p[0] for p in pts)
            y2 = max(p[1] for p in pts)
            label = f"#{pos} TOP OF STACK  {doc.digits}" if top else f"#{pos}  {doc.digits}"
            self._draw_tag(frame, label, (x1, y2 + 34), color)  # below the box

    def _draw_direction(self, frame, groups: List[DocumentGroup], scale: float):
        """Point a bold arrow at the first (top-of-stack) document, and draw faint
        arrows through the rest so the pick-up order 1 -> 2 -> 3 is obvious."""
        import cv2

        ordered = sorted([g for g in groups if g.stack_position],
                         key=lambda d: d.stack_position)
        if not ordered:
            return
        h, w = frame.shape[:2]
        centers = [(int(d.center[0] * scale), int(d.center[1] * scale)) for d in ordered]

        # Faint chain showing the order through the stack.
        for a, b in zip(centers, centers[1:]):
            cv2.arrowedLine(frame, a, b, (0, 200, 255), 2, cv2.LINE_AA, 0, 0.06)

        # Bold arrow into the first page: from just above its top edge, pointing down.
        first = ordered[0]
        fx = centers[0][0]
        x1, y1, x2, y2 = [int(v * scale) for v in first.region]
        if y1 > 100:                       # room above -> point down onto the page
            head, tail = (fx, y1), (fx, max(20, y1 - 80))
        else:                              # page near the top -> point up from below
            head, tail = (fx, min(h - 1, y2)), (fx, min(h - 1, y2 + 80))
        cv2.arrowedLine(frame, tail, head, (0, 220, 0), 6, cv2.LINE_AA, 0, 0.35)
        self._draw_tag(frame, "TAKE THIS FIRST (TOP)", (fx - 40, tail[1] - 4), (0, 220, 0))

    def _draw_stack_panel(self, frame, groups: List[DocumentGroup]):
        """Left-side panel: the stack top→bottom plus a plain instruction line."""
        import cv2

        ordered = sorted(groups, key=lambda d: d.stack_position or 0)
        lines = ["STACK (top -> bottom):"]
        if len(ordered) < 2:
            lines.append("show 2+ documents")
        else:
            for doc in ordered:
                suffix = "  <- put on TOP" if doc.stack_position == 1 else ""
                lines.append(f"{doc.stack_position}. {doc.digits}{suffix}")

        panel_w = 300
        panel_h = 24 * len(lines) + 16
        x0, y0 = 12, 70  # below the focus meter (top-left)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        y = y0 + 26
        for i, line in enumerate(lines):
            color = (255, 255, 255) if i == 0 else ((0, 220, 0) if i == 1 and len(ordered) >= 2
                                                    else (0, 165, 255))
            cv2.putText(frame, line, (x0 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            y += 24

        instr = stack_instruction(groups)
        if instr:
            cv2.putText(frame, instr, (x0, y0 + panel_h + 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2, cv2.LINE_AA)

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
