"""
Stretch goal (proposal §4): OCR on board/slide crops.

For each sampled frame the top ``board_crop_ratio`` of the image is cropped
(where the classroom board typically appears) and passed through Tesseract OCR.
The returned text is appended to the dialogue context window before Qwen fusion,
giving the model a three-way signal: student affect, teacher speech, board text.

Requirements:
    pip install pytesseract
    # system Tesseract: apt-get install tesseract-ocr  (Linux)
    #                   brew install tesseract           (macOS)
"""

from __future__ import annotations

import re
from pathlib import Path


def ocr_frame(
    image_path: str | Path,
    board_crop_ratio: float = 0.65,
) -> str:
    """
    Run Tesseract OCR on the upper ``board_crop_ratio`` of ``image_path``.

    Parameters
    ----------
    image_path:
        Path to a JPEG/PNG frame extracted from the video.
    board_crop_ratio:
        Fraction of the frame height to keep from the top (the board region).
        0.65 works well for most TIMSS-style classroom recordings where the
        board occupies the upper two-thirds of the frame.

    Returns
    -------
    Cleaned OCR string, or ``""`` on failure / no text found.
    """
    try:
        import cv2
        import pytesseract
    except ImportError as exc:
        raise RuntimeError(
            "Install pytesseract and OpenCV, and the Tesseract system binary, "
            "to enable OCR: pip install pytesseract opencv-python-headless"
        ) from exc

    img = cv2.imread(str(image_path))
    if img is None:
        return ""

    h = img.shape[0]
    crop_h = max(1, int(h * board_crop_ratio))
    board_region = img[:crop_h, :]

    # Tesseract works better on grayscale
    gray = cv2.cvtColor(board_region, cv2.COLOR_BGR2GRAY)

    raw = pytesseract.image_to_string(gray, config="--psm 6")
    return _clean_ocr(raw)


def ocr_frames(
    frame_paths: list[str | Path],
    board_crop_ratio: float = 0.65,
) -> str:
    """
    Run OCR on a list of frames and return deduplicated, combined text.

    Duplicate lines across frames are removed so the context window does not
    bloat when the board content changes slowly.
    """
    seen: set[str] = set()
    parts: list[str] = []
    for path in frame_paths:
        try:
            text = ocr_frame(path, board_crop_ratio=board_crop_ratio)
        except Exception:
            continue
        for line in text.splitlines():
            line = line.strip()
            if line and line not in seen:
                seen.add(line)
                parts.append(line)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINE_RE = re.compile(r"\n{3,}")


def _clean_ocr(raw: str) -> str:
    """Collapse whitespace and strip Tesseract noise."""
    lines = []
    for line in raw.splitlines():
        line = _WHITESPACE_RE.sub(" ", line).strip()
        # Drop very short lines that are usually OCR noise
        if len(line) >= 3:
            lines.append(line)
    text = "\n".join(lines)
    text = _BLANK_LINE_RE.sub("\n\n", text)
    return text.strip()
