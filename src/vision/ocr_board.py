"""
Stretch goal (proposal §4): OCR on board/slide crops.

Install optional dependency ``pytesseract`` and system Tesseract OCR, then call
``ocr_region`` on cropped arrays before fusion to append text to the context.
"""

from __future__ import annotations

from typing import Any


def ocr_region(image: Any) -> str:
    try:
        import pytesseract
    except ImportError as e:
        raise RuntimeError("Install pytesseract and Tesseract to enable OCR.") from e
    return pytesseract.image_to_string(image) or ""
