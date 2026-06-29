"""Turn raw OCR detections into a list of accepted bank account numbers.

Pure logic (no OpenCV / OCR engine) so it can be unit-tested directly.

Strategy (line-level)
----------------------
1. Group the OCR boxes into text lines (by vertical overlap).
2. A line is an ACCOUNT line only if it contains a full accepted label phrase —
   a number word + חשבון, e.g. "מספר חשבון" / "מס חשבון" / "מספר חשבון ראשי".
   A bare "חשבון" (e.g. "סוג חשבון") or a bare "מס" does NOT qualify.
3. On an account line, take only the number on the label's value side — for Hebrew
   RTL that is the number AFTER the label, i.e. to its left. Everything else on the
   line (other words, dashes, descriptions) is ignored.
4. Optionally (accept_unlabeled) accept any well-formed number on a line with no label.
5. De-duplicate by digit string, keeping the highest-confidence detection.
"""
from __future__ import annotations

from typing import List, Optional

from .config import Settings
from .types import AccountNumber, Detection
from .text_utils import (
    contains_account_word,
    extract_number_candidates,
    line_has_number_word,
    line_label_match,
)


def _vertical_overlap_ratio(a: Detection, b: Detection) -> float:
    _, ay1, _, ay2 = a.bbox
    _, by1, _, by2 = b.bbox
    inter = max(0.0, min(ay2, by2) - max(ay1, by1))
    smaller = max(1e-6, min(ay2 - ay1, by2 - by1))
    return inter / smaller


def _on_value_side(number: Detection, label: Detection, settings: Settings) -> bool:
    """Is `number` on the side of `label` where the account value belongs?

    On a Hebrew (RTL) line the number comes *after* the label word — to its left.
    Only numbers on that side are the account value; anything on the other side is
    ignored. `value_side="any"` disables the check.
    """
    side = settings.value_side
    if side == "any":
        return True
    if side == "right":
        return number.center[0] > label.center[0]
    return number.center[0] < label.center[0]  # default "left" (RTL)


def _group_into_lines(detections: List[Detection], settings: Settings) -> List[List[Detection]]:
    """Greedily group boxes that vertically overlap into the same text line."""
    lines: List[dict] = []
    for det in sorted(detections, key=lambda d: d.center[1]):
        _, y1, _, y2 = det.bbox
        target = None
        for line in lines:
            inter = max(0.0, min(y2, line["y2"]) - max(y1, line["y1"]))
            smaller = max(1e-6, min(y2 - y1, line["y2"] - line["y1"]))
            if inter / smaller >= settings.line_overlap_ratio:
                target = line
                break
        if target is None:
            lines.append({"dets": [det], "y1": y1, "y2": y2})
        else:
            target["dets"].append(det)
            target["y1"] = min(target["y1"], y1)
            target["y2"] = max(target["y2"], y2)
    return [line["dets"] for line in lines]


def _account_word_box(line: List[Detection], settings: Settings) -> Optional[Detection]:
    """The box on this line holding the account noun (חשבון) — the value-side anchor."""
    best: Optional[Detection] = None
    for det in line:
        if contains_account_word(det.text, settings):
            if best is None or det.confidence > best.confidence:
                best = det
    return best


def associate_accounts(detections: List[Detection], settings: Settings) -> List[AccountNumber]:
    """Return the bank account numbers found among `detections`."""
    accepted: List[AccountNumber] = []
    for line in _group_into_lines(detections, settings):
        texts = [d.text for d in line]
        matched = line_label_match(texts, settings)
        if matched is not None and settings.require_number_word and not line_has_number_word(texts, settings):
            matched = None  # extra-strict mode: phrase also needs its "מספר"/"מס" word

        if matched is None:
            if settings.accept_unlabeled:
                for det in line:
                    if det.confidence < settings.min_ocr_confidence:
                        continue
                    for digits in extract_number_candidates(det.text, settings):
                        accepted.append(AccountNumber(digits=digits, detection=det,
                                                      label_text=None, confidence=det.confidence))
            continue

        anchor = _account_word_box(line, settings)
        for det in line:
            if det.confidence < settings.min_ocr_confidence:
                continue
            for digits in extract_number_candidates(det.text, settings):
                # Accept a number that sits after the label (value side), or that
                # shares the label's own box (e.g. "מספר חשבון 299868").
                if anchor is None or det is anchor or _on_value_side(det, anchor, settings):
                    accepted.append(AccountNumber(digits=digits, detection=det,
                                                  label_text=matched, confidence=det.confidence))

    return _dedupe(accepted)


def _dedupe(accounts: List[AccountNumber]) -> List[AccountNumber]:
    """Keep one entry per digit string — the most confident one."""
    best: dict = {}
    for acc in accounts:
        cur = best.get(acc.digits)
        if cur is None or acc.confidence > cur.confidence:
            best[acc.digits] = acc
    return list(best.values())
