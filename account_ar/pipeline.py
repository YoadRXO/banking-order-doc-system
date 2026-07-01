"""Background OCR worker: processes the latest camera frame and publishes the
most recent ranked-account result for the UI to draw.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Tuple

from .config import Settings
from .detector import associate_accounts
from .grouping import group_documents
from .ocr_engine import OcrEngine
from .ordering import rank_accounts
from .types import AccountNumber, FrameResult


class AccountTracker:
    """Remember accounts across frames so a single readable frame "sticks".

    Cheap webcam video of a faint, hand-held document yields mostly unreadable
    frames (glare, blur, motion) with the occasional good one. Without memory the
    overlay flickers empty almost always; with it, one clean read keeps the
    account on screen for `ttl` seconds, refreshing each time it is re-seen.
    """

    def __init__(self, ttl: float, confirm_sightings: int = 1, lock: bool = False):
        self.ttl = ttl
        self.confirm_sightings = max(1, confirm_sightings)
        # lock=True: once an instance is confirmed it is "locked" — frozen at its found
        # position and never expired, so found numbers ACCUMULATE while the user pans to
        # find more (instead of refreshing away). clear() resets for the next batch.
        self.lock = lock
        # Two separate jobs, kept apart so neither breaks the other:
        #  * confirmation is by VALUE (robust to box jitter between frames) — a number is
        #    trusted once its digits have been read >= confirm_sightings times.
        #  * display is by INSTANCE (position) — the same number on two papers shows as
        #    two boxes; frame-to-frame jitter of one paper is matched by proximity.
        self._values: Dict[str, dict] = {}     # digits -> {"count", "last"}
        self._instances: List[dict] = []        # [{"acc", "last", "locked"}]

    def _nearest_instance(self, acc: AccountNumber):
        best, best_d = None, None
        cx, cy = acc.detection.center
        tol = 1.5 * max(acc.detection.width, acc.detection.height, 1.0)
        for it in self._instances:
            if it["acc"].digits != acc.digits:
                continue
            ix, iy = it["acc"].detection.center
            d = ((cx - ix) ** 2 + (cy - iy) ** 2) ** 0.5
            if d <= tol and (best_d is None or d < best_d):
                best, best_d = it, d
        return best

    def update(self, accounts: List[AccountNumber], now: Optional[float] = None) -> List[AccountNumber]:
        now = time.time() if now is None else now
        for acc in accounts:
            v = self._values.get(acc.digits)
            self._values[acc.digits] = {"count": (v["count"] + 1 if v else 1), "last": now}
            match = self._nearest_instance(acc)
            if match is not None:
                if not match.get("locked"):   # a locked box stays frozen where it was found
                    match["acc"] = acc        # newest detection (freshest box) wins
                match["last"] = now
            else:
                self._instances.append({"acc": acc, "last": now, "locked": False})
        confirmed = {d for d, v in self._values.items() if v["count"] >= self.confirm_sightings}
        # In lock mode a confirmed instance becomes permanent (frozen, never expires).
        if self.lock:
            for it in self._instances:
                if it["acc"].digits in confirmed:
                    it["locked"] = True
        # Expire: locked instances persist; everything else drops after ttl.
        self._values = {d: v for d, v in self._values.items()
                        if now - v["last"] <= self.ttl
                        or (self.lock and v["count"] >= self.confirm_sightings)}
        self._instances = [it for it in self._instances
                           if it.get("locked") or now - it["last"] <= self.ttl]
        # SAFETY: only surface instances whose VALUE was read >= confirm_sightings times
        # (or that are already locked) — a single rotated/glare misread never shows.
        return [it["acc"] for it in self._instances
                if it.get("locked") or it["acc"].digits in confirmed]

    def clear(self) -> None:
        self._values.clear()
        self._instances.clear()


class ARPipeline:
    def __init__(self, settings: Settings, camera):
        self.settings = settings
        self.camera = camera
        self.ocr = OcrEngine(settings)
        self._result = FrameResult()
        self._tracker = AccountTracker(settings.track_seconds, settings.confirm_sightings,
                                       settings.lock_found)
        self._lock = threading.Lock()
        self._running = False
        self._paused = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def roi_rect(self, frame):
        """Central magnifier box (x, y, w, h) in `frame` pixels, or None when disabled."""
        if not self.settings.roi_enabled:
            return None
        h, w = frame.shape[:2]
        rw = int(w * self.settings.roi_width_frac)
        rh = int(h * self.settings.roi_height_frac)
        return ((w - rw) // 2, (h - rh) // 2, rw, rh)

    @staticmethod
    def _shift(det: "Detection", dx: int, dy: int) -> "Detection":
        from .types import Detection

        return Detection(text=det.text, confidence=det.confidence,
                         polygon=[(x + dx, y + dy) for (x, y) in det.polygon])

    def _sharpness(self, frame) -> float:
        """Normalized focus score (variance of the Laplacian) of `frame`.

        Resized to a fixed width first so the number means the same thing on any
        camera resolution. Higher = sharper; a badly out-of-focus or motion-blurred
        frame scores near zero.
        """
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        nw = self.settings.sharpness_norm_width
        if w > nw:
            gray = cv2.resize(gray, (nw, int(h * nw / w)), interpolation=cv2.INTER_AREA)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _loop(self) -> None:
        self.ocr.ensure_loaded()  # verify the Tesseract engine once
        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.02)
                continue
            try:
                size = (frame.shape[1], frame.shape[0])
                # When the magnifier box is on, work on just that central region so the
                # focus check and OCR both apply to what the user is aiming at.
                roi = self.roi_rect(frame)
                if roi is not None:
                    rx, ry, rw, rh = roi
                    region = frame[ry:ry + rh, rx:rx + rw]
                else:
                    rx, ry, region = 0, 0, frame

                sharpness = self._sharpness(region)
                # Too blurry to read anything — don't waste 1-2s OCRing noise. Keep any
                # recently-tracked accounts on screen and tell the UI the frame was skipped.
                if sharpness < self.settings.min_sharpness:
                    tracked = (self._tracker.update([]) if self.settings.track_seconds > 0
                               else [])
                    ranked = rank_accounts(tracked, self.settings)
                    pages = self._detect_pages(region, rx, ry)
                    with self._lock:
                        self._result = FrameResult(
                            accounts=ranked, frame_size=size, ocr_ms=0.0, detections=[],
                            sharpness=sharpness, ocr_skipped=True,
                            groups=self._grouped(ranked, pages))
                    time.sleep(0.01)
                    continue

                detections, ocr_ms, _ = self.ocr.read(region)
                if roi is not None:  # map ROI-local boxes back onto the full frame
                    detections = [self._shift(d, rx, ry) for d in detections]
                accounts = associate_accounts(detections, self.settings)
                if self.settings.track_seconds > 0:
                    accounts = self._tracker.update(accounts)
                ranked = rank_accounts(accounts, self.settings)
                pages = self._detect_pages(region, rx, ry)
                with self._lock:
                    self._result = FrameResult(accounts=ranked, frame_size=size,
                                               ocr_ms=ocr_ms, detections=detections,
                                               sharpness=sharpness, ocr_skipped=False,
                                               groups=self._grouped(ranked, pages))
            except Exception as exc:  # keep the worker alive on transient errors
                print(f"[pipeline] OCR error: {exc}")
                time.sleep(0.1)

    def _detect_pages(self, region, rx: int, ry: int):
        """Detect paper rectangles in `region`, mapped back onto the full frame.

        Only runs in stacking mode with page detection on; returns [] otherwise (so the
        grouping falls back to number-only clustering).
        """
        if not (self.settings.stack_order_enabled and self.settings.detect_pages):
            return []
        from .page_detect import detect_pages

        pages = detect_pages(region, self.settings)
        if rx or ry:  # shift ROI-local page boxes onto the full frame
            pages = [[(x + rx, y + ry) for (x, y) in quad] for quad in pages]
        return pages

    def _grouped(self, ranked: List[AccountNumber], pages=None):
        """Stacking order for the current accounts (multi-document mode), else []."""
        if not self.settings.stack_order_enabled:
            return []
        return group_documents(ranked, self.settings, pages=pages)

    @property
    def ocr_ready(self) -> bool:
        return self.ocr.ready

    def toggle_pause(self) -> bool:
        self._paused = not self._paused
        return self._paused

    def clear_tracked(self) -> None:
        """Forget remembered accounts (e.g. after moving to a different document)."""
        self._tracker.clear()

    def toggle_lock(self) -> bool:
        """Toggle lock/accumulate mode: found numbers freeze in place instead of expiring."""
        self._tracker.lock = not self._tracker.lock
        return self._tracker.lock

    def get_result(self) -> FrameResult:
        with self._lock:
            return self._result

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
