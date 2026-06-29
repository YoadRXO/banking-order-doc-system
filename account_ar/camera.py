"""Threaded webcam capture so frame grabbing never blocks the UI loop."""
from __future__ import annotations

import threading
from typing import Optional


class Camera:
    def __init__(self, index: int = 0, width: int = 0, height: int = 0):
        self.index = index
        self.width = width
        self.height = height
        self._cap = None
        self._lock = threading.Lock()
        self._frame = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def open(self) -> None:
        import cv2

        # CAP_DSHOW avoids slow MSMF startup on Windows; harmless elsewhere.
        backend = getattr(cv2, "CAP_DSHOW", 0)
        self._cap = cv2.VideoCapture(self.index, backend)
        if not self._cap or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.index)
        if not self._cap or not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.index}")

        # Request a higher capture resolution so document text is readable.
        if self.width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame = frame

    def read(self):
        """Return the most recent frame (a copy), or None if not ready yet."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def release(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
