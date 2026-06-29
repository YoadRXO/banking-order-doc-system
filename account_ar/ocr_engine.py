"""Tesseract OCR wrapper (Hebrew + English).

Uses pytesseract with the `heb` language data. Tesseract is CPU-only but fast
enough for document OCR, needs no Docker/GPU stack, and packages cleanly into a
standalone .exe.

Requires the Tesseract ENGINE installed with the Hebrew language pack:
  Windows: UB-Mannheim installer (tick "Hebrew")
  Linux:   sudo apt install tesseract-ocr tesseract-ocr-heb
"""
from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple

from .config import Settings, default_tessdata_dir
from .types import Detection


def _candidate_tesseract_paths() -> List[str]:
    """Common Windows install locations for tesseract.exe."""
    paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    local = os.environ.get("LOCALAPPDATA")
    if local:  # per-user installs
        paths.append(os.path.join(local, "Tesseract-OCR", "tesseract.exe"))
        paths.append(os.path.join(local, "Programs", "Tesseract-OCR", "tesseract.exe"))
    return paths


def find_tesseract_cmd() -> Optional[str]:
    """Return the path to tesseract.exe if found in a known location, else None."""
    for candidate in _candidate_tesseract_paths():
        if os.path.isfile(candidate):
            return candidate
    return None


class OcrEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._ready = False

    def _locate_tesseract(self) -> None:
        import pytesseract

        cmd = self.settings.tesseract_cmd or find_tesseract_cmd()
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd

    def ensure_loaded(self) -> None:
        if self._ready:
            return
        import pytesseract

        self._locate_tesseract()
        try:
            pytesseract.get_tesseract_version()
        except Exception as exc:  # tesseract.exe missing / not on PATH
            raise RuntimeError(
                "Tesseract engine not found. Install it (Windows: UB-Mannheim "
                "installer, tick 'Hebrew'; Linux: sudo apt install tesseract-ocr "
                "tesseract-ocr-heb), or set 'tesseract_cmd' in config.json to the "
                f"full path of tesseract.exe.\nOriginal error: {exc}"
            )
        self._ready = True

    @property
    def ready(self) -> bool:
        return self._ready

    def _preprocess(self, bgr, cv2):
        """Clean the image before OCR.

        `clahe` (default) applies local contrast equalization — the single biggest
        win for cheap webcams photographing faint Hebrew bank print in poor light.
        It stays grayscale (not binarized) so Tesseract sees smooth strokes instead
        of broken, speckled glyphs. Thresholding modes (`otsu`/`adaptive`) suit
        crisp, high-contrast scans but tend to erase faint print.
        """
        mode = self.settings.preprocess
        if mode == "none":
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if mode == "clahe":
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            return clahe.apply(gray)
        if mode == "otsu":
            _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif mode == "adaptive":
            gray = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
            )
        return gray  # "gray" returns plain grayscale; pytesseract accepts 2-D arrays

    def _config_string(self) -> str:
        parts = [f"--oem {self.settings.tesseract_oem}", f"--psm {self.settings.tesseract_psm}"]
        tessdata = self.settings.tessdata_dir or default_tessdata_dir()
        if tessdata:
            parts.append(f'--tessdata-dir "{tessdata}"')
        return " ".join(parts)

    def read(self, frame_bgr) -> Tuple[List[Detection], float, Tuple[int, int]]:
        """Run OCR on a BGR frame.

        Returns (detections, elapsed_ms, frame_size). One Detection per word, in
        original frame coordinates.
        """
        import cv2
        import pytesseract
        from pytesseract import Output

        self.ensure_loaded()

        h, w = frame_bgr.shape[:2]
        scale = 1.0
        work = frame_bgr
        if w > self.settings.ocr_max_width:
            scale = self.settings.ocr_max_width / float(w)
            work = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_AREA)
        elif w < self.settings.ocr_min_width:
            # Upscale small frames so text is tall enough for Tesseract (~30px caps).
            scale = self.settings.ocr_min_width / float(w)
            work = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_CUBIC)

        proc = self._preprocess(work, cv2)
        lang = "+".join(self.settings.languages)

        start = time.perf_counter()
        data = pytesseract.image_to_data(
            proc, lang=lang, config=self._config_string(), output_type=Output.DICT
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        detections: List[Detection] = []
        n = len(data["text"])
        for i in range(n):
            text = (data["text"][i] or "").strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if conf < 0:
                continue
            x = data["left"][i] / scale
            y = data["top"][i] / scale
            bw = data["width"][i] / scale
            bh = data["height"][i] / scale
            polygon = [(x, y), (x + bw, y), (x + bw, y + bh), (x, y + bh)]
            detections.append(Detection(text=text, confidence=conf / 100.0, polygon=polygon))

        return detections, elapsed_ms, (w, h)
