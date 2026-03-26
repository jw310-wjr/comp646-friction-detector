"""
Dialogue strategy tagging.

The proposal uses Edu-ConvoKit (Talk Moves + uptake) and Tutor-CoPilot RoBERTa
classifiers. Those models live in separate releases; this module provides a
deterministic heuristic layer so the end-to-end pipeline runs out of the box,
and a small hook structure if you wire true classifiers later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from schemas import AnnotatedUtterance, Quality, TranscriptSegment

_LOW_QUALITY_PATTERNS = [
    r"\b(the answer is|here'?s the answer|it'?s \d+)\b",
    r"\b(let me (just )?(tell|show) you)\b",
    r"\byou (should |just )(use|plug|multiply|divide)\b",
    r"\bthis (always|never) works\b",
    r"\bthe formula is\b",
]

_HIGH_PRESSURE_PATTERNS = [
    r"\bare you sure\b",
    r"\bwhy did you think that\b",
    r"\bthat'?s wrong\b",
    r"\bno[, ]+(that|this)\b",
]


@dataclass
class HeuristicStrategyAnnotator:
    """Keyword heuristics standing in for pre-trained classifiers."""

    def annotate(self, segments: list[TranscriptSegment]) -> list[AnnotatedUtterance]:
        out: list[AnnotatedUtterance] = []
        for s in segments:
            text_lower = s.text.lower()
            low = any(re.search(p, text_lower) for p in _LOW_QUALITY_PATTERNS)
            pressure = any(re.search(p, text_lower) for p in _HIGH_PRESSURE_PATTERNS)
            if low:
                quality: Quality = "low"
            elif pressure:
                quality = "low"
            elif len(s.text.split()) <= 3 and "?" not in s.text:
                quality = "unknown"
            else:
                quality = "high"
            move = "Pressing for Accuracy" if pressure else ("No Move" if not s.text else "Other")
            if "?" in s.text and quality != "low":
                move = "Pressing for Reasoning"
            out.append(
                AnnotatedUtterance(
                    t_start=s.t_start,
                    t_end=s.t_end,
                    text=s.text,
                    talk_move=move,
                    strategy_quality=quality,
                    uptake_score=None,
                    notes="heuristic; replace with Edu-ConvoKit + Tutor-CoPilot",
                )
            )
        return out
