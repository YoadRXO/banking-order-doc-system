"""Find the document *pages* (sheets of paper) in a camera frame with OpenCV.

Used by the multi-document stacking feature: instead of only clustering the account
numbers, we detect each physical page as a bright rectangle so every number can be tied
to the page it sits on, the pages can be outlined, and a direction arrow can point at the
one that goes on top.

Heuristic (papers are bright rectangles on a darker desk):
  gray -> blur -> Otsu threshold (bright regions) -> close gaps -> external contours
  -> keep blobs that are a sensible fraction of the frame and roughly page-shaped.

Returns axis-aligned page boxes as 4-point polygons (clockwise from top-left), in the
coordinate space of the frame passed in. Robust to slight tilt (bounding box covers it)
and degrades to "no pages found" -> the caller falls back to number-only clustering.
"""
from __future__ import annotations

from typing import List, Tuple

from .config import Settings
from .types import Point


def _boxes_overlap_frac(a, b) -> float:
    """Intersection area over the smaller box's area (how much the two boxes coincide)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    smaller = max(1.0, min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1)))
    return inter / smaller


def _dedupe(boxes: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
    """Drop boxes that sit (mostly) inside a bigger one — keep the largest of each overlap."""
    ordered = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    kept: List[Tuple[float, float, float, float]] = []
    for b in ordered:
        if any(_boxes_overlap_frac(b, k) >= 0.6 for k in kept):
            continue
        kept.append(b)
    return kept


def detect_pages(frame, settings: Settings) -> List[List[Point]]:
    """Return detected page rectangles as 4-point polygons, largest first."""
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    frame_area = float(max(1, w * h))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Bright paper vs. darker background. Otsu picks the split automatically.
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Close text/line holes so each sheet is one solid blob.
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2)

    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[Tuple[float, float, float, float]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (settings.page_min_area_frac * frame_area <= area
                <= settings.page_max_area_frac * frame_area):
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / float(bh) if bh else 0.0
        if aspect < 0.2 or aspect > 5.0:      # reject thin strips (not a page)
            continue
        if bw * bh < 1.3 * area:              # bounding box ~ contour => reasonably rectangular
            boxes.append((float(x), float(y), float(x + bw), float(y + bh)))

    pages: List[List[Point]] = []
    for (x1, y1, x2, y2) in _dedupe(boxes):
        pages.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    return pages
