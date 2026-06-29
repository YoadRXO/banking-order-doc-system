"""Bank Account AR Detector.

Detects Hebrew-labelled bank account numbers on a printed page via webcam and
overlays wireframe boxes annotated with their correct sort order.

Only the lightweight, dependency-free modules are imported eagerly here so that
the pure-logic test suite can import the package without OpenCV / pytesseract
being installed.
"""

__version__ = "0.1.0"
