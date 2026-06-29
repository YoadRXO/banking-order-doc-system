"""Entry point: open the camera, run the OCR pipeline, draw the AR overlay.

Keys:
  q / ESC  quit
  space    save a screenshot of the current view
  p        pause / resume OCR processing
  a        toggle "accept unlabeled numbers" mode
"""
from __future__ import annotations

import argparse
import os
import time

from .camera import Camera
from .config import Settings
from .overlay import Overlay
from .pipeline import ARPipeline


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bank Account AR Detector")
    p.add_argument("--camera", type=int, default=None, help="camera index (default from config)")
    p.add_argument("--config", type=str, default=None, help="path to config.json")
    p.add_argument("--accept-unlabeled", action="store_true",
                   help="accept numbers even without a nearby Hebrew label")
    p.add_argument("--image", type=str, default=None,
                   help="process a single image file instead of the webcam")
    p.add_argument("--output", type=str, default=None,
                   help="where to save the annotated image (with --image)")
    p.add_argument("--show", action="store_true",
                   help="also open a window in --image mode (needs a display)")
    p.add_argument("--debug", action="store_true",
                   help="print every raw OCR detection (diagnostics)")
    p.add_argument("--preprocess", choices=["none", "gray", "clahe", "otsu", "adaptive"], default=None,
                   help="image cleanup before OCR (default from config: clahe)")
    return p.parse_args(argv)


def _annotated_path(image_path: str) -> str:
    root, ext = os.path.splitext(image_path)
    return f"{root}_annotated{ext or '.png'}"


def run_image(settings: Settings, image_path: str, output_path=None,
              show: bool = False, debug: bool = False) -> int:
    """Process one still image: OCR -> detect -> order -> print + save annotated copy."""
    import cv2

    from .detector import associate_accounts
    from .ocr_engine import OcrEngine
    from .ordering import rank_accounts
    from .overlay import Overlay

    if not os.path.isfile(image_path):
        print(f"[error] image not found: {image_path}")
        return 2
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[error] could not read image: {image_path}")
        return 2

    print("Initializing OCR (Tesseract)...")
    ocr = OcrEngine(settings)
    detections, ocr_ms, size = ocr.read(frame)

    print(f"\nOCR found {len(detections)} text boxes in {ocr_ms:.0f} ms.")
    if debug:
        from .text_utils import is_label, extract_number_candidates
        print("--- raw OCR detections ---")
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            tags = []
            if is_label(d.text, settings):
                tags.append("LABEL")
            nums = extract_number_candidates(d.text, settings)
            if nums:
                tags.append(f"NUM{nums}")
            print(f"  conf={d.confidence:.2f} box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) "
                  f"text={d.text!r} {' '.join(tags)}")
        print("--------------------------")

    accounts = rank_accounts(associate_accounts(detections, settings), settings)

    print(f"Accepted {len(accounts)} account number(s):")
    for a in accounts:
        print(f"  #{a.rank}  {a.digits}   (label={a.label_text!r}, conf={a.confidence:.2f})")
    if not accounts:
        print("  (none — try --accept-unlabeled, or check the image / config.json keywords)")

    overlay = Overlay()
    scale = frame.shape[1] / float(size[0]) if size[0] else 1.0
    overlay.draw(frame, accounts, status=f"{len(accounts)} account(s)", scale=scale)

    out = output_path or _annotated_path(image_path)
    cv2.imwrite(out, frame)
    print(f"\nSaved annotated image: {os.path.abspath(out)}")

    if show:
        try:
            cv2.imshow(settings.window_name, frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as exc:
            print(f"[info] could not open a window ({exc}); open the saved file instead.")
    return 0


def run(settings: Settings, debug: bool = False) -> int:
    import cv2

    camera = Camera(settings.camera_index, settings.camera_width, settings.camera_height)
    try:
        camera.open()
    except RuntimeError as exc:
        print(f"[error] {exc}")
        return 2

    pipeline = ARPipeline(settings, camera)
    pipeline.start()
    overlay = Overlay()

    print("Initializing OCR (Tesseract)...")
    print("Keys:  q/ESC quit | space screenshot | p pause | a accept-unlabeled | "
          "b preprocess-mode | c clear-memory")

    last = time.perf_counter()
    fps = 0.0
    try:
        while True:
            frame = camera.read()
            if frame is None:
                if cv2.waitKey(10) & 0xFF in (ord("q"), 27):
                    break
                continue

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - last))
            last = now

            result = pipeline.get_result()
            scale = 1.0
            if result.frame_size[0]:
                scale = frame.shape[1] / float(result.frame_size[0])

            if not pipeline.ocr_ready:
                status = "Loading OCR model..."
            else:
                status = (f"FPS {fps:4.1f} | OCR {result.ocr_ms:5.0f}ms | "
                          f"{len(result.accounts)} acct | prep={settings.preprocess} | "
                          f"unlabeled={'ON' if settings.accept_unlabeled else 'off'}")

            raw = frame.copy()  # clean frame (no overlay) for screenshots / OCR debug
            overlay.draw(frame, result.accounts, status=status, scale=scale)
            cv2.imshow(settings.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("p"):
                paused = pipeline.toggle_pause()
                print(f"[ui] processing {'paused' if paused else 'resumed'}")
            elif key == ord("a"):
                settings.accept_unlabeled = not settings.accept_unlabeled
                print(f"[ui] accept_unlabeled = {settings.accept_unlabeled}")
            elif key == ord("b"):
                modes = ["clahe", "none", "gray", "otsu", "adaptive"]
                cur = modes.index(settings.preprocess) if settings.preprocess in modes else 0
                settings.preprocess = modes[(cur + 1) % len(modes)]
                print(f"[ui] preprocess = {settings.preprocess}")
            elif key == ord("c"):
                pipeline.clear_tracked()
                print("[ui] cleared remembered accounts")
            elif key == ord(" "):
                ts = int(time.time())
                cv2.imwrite(os.path.abspath(f"screenshot_{ts}.png"), frame)
                cv2.imwrite(os.path.abspath(f"raw_{ts}.png"), raw)  # clean, for --image --debug
                print(f"[ui] saved screenshot_{ts}.png + raw_{ts}.png "
                      f"({frame.shape[1]}x{frame.shape[0]})")
                if debug:
                    print(f"[debug] {len(result.detections)} OCR detections this frame:")
                    for d in result.detections:
                        print(f"   conf={d.confidence:.2f} text={d.text!r}")
    finally:
        pipeline.stop()
        camera.release()
        cv2.destroyAllWindows()
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    settings = Settings.load(args.config)
    if args.camera is not None:
        settings.camera_index = args.camera
    if args.accept_unlabeled:
        settings.accept_unlabeled = True
    if args.preprocess is not None:
        settings.preprocess = args.preprocess
    if args.image:
        return run_image(settings, args.image, args.output, args.show, args.debug)
    return run(settings, args.debug)


if __name__ == "__main__":
    raise SystemExit(main())
