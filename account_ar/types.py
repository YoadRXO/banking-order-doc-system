"""Core data types shared across the pipeline.

Intentionally dependency-free (stdlib only) so detection/ordering logic can be
unit-tested without OpenCV or the OCR engine (Tesseract/pytesseract) installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Point = Tuple[float, float]


@dataclass
class Detection:
    """A single OCR text box.

    `polygon` is the 4-point quad returned by the OCR engine (clockwise from top-left).
    Axis-aligned helpers are derived from it for the association math.
    """

    text: str
    confidence: float
    polygon: List[Point]

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Axis-aligned bounding box as (x1, y1, x2, y2)."""
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def width(self) -> float:
        x1, _, x2, _ = self.bbox
        return x2 - x1

    @property
    def height(self) -> float:
        _, y1, _, y2 = self.bbox
        return y2 - y1

    @property
    def center(self) -> Point:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def overlaps(self, other: "Detection", frac: float = 0.3) -> bool:
        """True if this box overlaps `other` by >= `frac` of the smaller box's area.

        Used to tell whether two same-value reads are the *same* physical number
        (overlapping) or two different papers showing the same number (disjoint).
        """
        ax1, ay1, ax2, ay2 = self.bbox
        bx1, by1, bx2, by2 = other.bbox
        ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        iy = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = ix * iy
        if inter <= 0:
            return False
        smaller = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
        return inter >= frac * max(1.0, smaller)


@dataclass
class AccountNumber:
    """An accepted bank account number plus where it was found."""

    digits: str
    detection: Detection
    label_text: Optional[str] = None
    confidence: float = 0.0
    rank: Optional[int] = None

    @property
    def value(self) -> int:
        return int(self.digits)


@dataclass
class DocumentGroup:
    """One physical document — a spatial cluster of account detections — plus its
    place in the physical stacking order.

    Used by the multi-document feature: show two (or more) separate papers at once,
    each with its own account number, and the app tells you which paper to put on top.
    `stack_position` is 1-based; **1 = TOP of the stack**.
    """

    accounts: List[AccountNumber]
    stack_position: Optional[int] = None
    page_quad: Optional[List[Point]] = None  # the detected paper rectangle, if a page was found

    @property
    def primary(self) -> AccountNumber:
        """The account that represents this document for ordering.

        Prefer a label-anchored read (a real "…חשבון" match) over a bare number, then
        the most confident — so a stray digit run doesn't outrank the real account.
        """
        return sorted(self.accounts, key=lambda a: (a.label_text is None, -a.confidence))[0]

    @property
    def digits(self) -> str:
        return self.primary.digits

    @property
    def value(self) -> int:
        return self.primary.value

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Axis-aligned box enclosing every detection in the document."""
        boxes = [a.detection.bbox for a in self.accounts]
        return (min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes))

    @property
    def region(self) -> Tuple[float, float, float, float]:
        """The box to outline/aim at: the detected page if known, else the numbers' box."""
        if self.page_quad:
            xs = [p[0] for p in self.page_quad]
            ys = [p[1] for p in self.page_quad]
            return (min(xs), min(ys), max(xs), max(ys))
        return self.bbox

    @property
    def center(self) -> Point:
        x1, y1, x2, y2 = self.region
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class FrameResult:
    """Everything the UI needs to draw one processed frame's worth of results."""

    accounts: List[AccountNumber] = field(default_factory=list)
    frame_size: Tuple[int, int] = (0, 0)  # (width, height) the OCR ran on
    ocr_ms: float = 0.0
    detections: List[Detection] = field(default_factory=list)  # raw OCR, for debug
    sharpness: float = 0.0                # normalized focus score of the last frame
    ocr_skipped: bool = False             # True when the frame was too blurry to OCR
    groups: List[DocumentGroup] = field(default_factory=list)  # stacking order (multi-doc mode)
