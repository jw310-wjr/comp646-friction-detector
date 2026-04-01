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
    # Additional patterns for TIMSS-style classroom discourse
    r"\bjust (copy|write|put|add|subtract|multiply|divide)\b",
    r"\b(remember|recall) (that |the |how )?(formula|rule|fact|step)",
    r"\bthe (rule|definition|property) (is|says)\b",
    r"\byou (need to|have to|must) (know|memorize|remember)\b",
    r"\b(it'?s|that'?s) (just|simply|only|basically)\b",
    r"\b(fill in|copy down|write down) (the|this|that)\b",
]

_HIGH_PRESSURE_PATTERNS = [
    r"\bare you sure\b",
    r"\bwhy did you think that\b",
    r"\bthat'?s wrong\b",
    r"\bno[, ]+(that|this)\b",
    # Additional high-pressure patterns
    r"\bno[,.]? (try again|incorrect|not right)\b",
    r"\b(wrong|incorrect)[,.]\b",
    r"\bwhy (would|did) you (say|think|do) that\b",
    r"\bdoes (anyone|somebody) (else )?know\b",
    r"\bcome on\b",
    r"\bpay attention\b",
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
