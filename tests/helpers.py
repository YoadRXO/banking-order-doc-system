"""Shared test helpers."""
from account_ar.types import Detection


def det(text: str, x1: float, y1: float, x2: float, y2: float, conf: float = 0.9) -> Detection:
    """Build a Detection with an axis-aligned rectangular polygon."""
    polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return Detection(text=text, confidence=conf, polygon=polygon)
