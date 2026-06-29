"""Background OCR worker: processes the latest camera frame and publishes the
most recent ranked-account result for the UI to draw.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Tuple

from .config import Settings
from .detector import associate_accounts
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

    def __init__(self, ttl: float):
        self.ttl = ttl
        self._seen: Dict[str, Tuple[AccountNumber, float]] = {}

    def update(self, accounts: List[AccountNumber], now: Optional[float] = None) -> List[AccountNumber]:
        now = time.time() if now is None else now
        for acc in accounts:
            self._seen[acc.digits] = (acc, now)  # newest detection (freshest box) wins
        self._seen = {d: (a, t) for d, (a, t) in self._seen.items() if now - t <= self.ttl}
        return [a for (a, _) in self._seen.values()]

    def clear(self) -> None:
        self._seen.clear()


class ARPipeline:
    def __init__(self, settings: Settings, camera):
        self.settings = settings
        self.camera = camera
        self.ocr = OcrEngine(settings)
        self._result = FrameResult()
        self._tracker = AccountTracker(settings.track_seconds)
        self._lock = threading.Lock()
        self._running = False
        self._paused = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

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
                detections, ocr_ms, size = self.ocr.read(frame)
                accounts = associate_accounts(detections, self.settings)
                if self.settings.track_seconds > 0:
                    accounts = self._tracker.update(accounts)
                ranked = rank_accounts(accounts, self.settings)
                with self._lock:
                    self._result = FrameResult(accounts=ranked, frame_size=size,
                                               ocr_ms=ocr_ms, detections=detections)
            except Exception as exc:  # keep the worker alive on transient errors
                print(f"[pipeline] OCR error: {exc}")
                time.sleep(0.1)

    @property
    def ocr_ready(self) -> bool:
        return self.ocr.ready

    def toggle_pause(self) -> bool:
        self._paused = not self._paused
        return self._paused

    def clear_tracked(self) -> None:
        """Forget remembered accounts (e.g. after moving to a different document)."""
        self._tracker.clear()

    def get_result(self) -> FrameResult:
        with self._lock:
            return self._result

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
