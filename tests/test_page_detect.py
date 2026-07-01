"""Page (paper-rectangle) detection — needs OpenCV, so skipped when it isn't installed."""
import unittest

try:
    import cv2  # noqa: F401
    import numpy as np
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

from account_ar.config import Settings


@unittest.skipUnless(HAVE_CV2, "OpenCV/numpy not installed")
class TestPageDetect(unittest.TestCase):
    def test_finds_two_sheets_on_a_dark_desk(self):
        from account_ar.page_detect import detect_pages

        # Dark "desk" with two bright "sheets" side by side.
        frame = np.zeros((720, 1280, 3), np.uint8)
        frame[150:600, 120:520] = 235     # left sheet
        frame[150:600, 760:1160] = 235    # right sheet
        pages = detect_pages(frame, Settings())

        self.assertEqual(len(pages), 2)
        # Each returned page is a 4-point box; their centres are on opposite frame halves.
        centres = sorted((sum(p[0] for p in q) / 4.0) for q in pages)
        self.assertLess(centres[0], 640)
        self.assertGreater(centres[1], 640)

    def test_blank_frame_finds_no_pages(self):
        from account_ar.page_detect import detect_pages

        frame = np.zeros((480, 640, 3), np.uint8)
        self.assertEqual(detect_pages(frame, Settings()), [])


if __name__ == "__main__":
    unittest.main()
