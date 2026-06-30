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
class FrameResult:
    """Everything the UI needs to draw one processed frame's worth of results."""

    accounts: List[AccountNumber] = field(default_factory=list)
    frame_size: Tuple[int, int] = (0, 0)  # (width, height) the OCR ran on
    ocr_ms: float = 0.0
    detections: List[Detection] = field(default_factory=list)  # raw OCR, for debug
    sharpness: float = 0.0                # normalized focus score of the last frame
    ocr_skipped: bool = False             # True when the frame was too blurry to OCR
