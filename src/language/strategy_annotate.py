"""
Dialogue strategy tagging — proposal §2.

Primary path: Edu-ConvoKit ELECTRA Talk Moves classifier
  Model: YaHi/teacher_electra_small (7-class, fine-tuned on TalkMoves dataset)
  Labels: No Move / Keeping Everyone Together / Relating to Another Student's
          Idea / Restating / Revoicing / Pressing for Accuracy / Pressing for
          Reasoning

Strategy quality (high / low / unknown) is derived from the Talk Move label.
The Tutor CoPilot RoBERTa quality classifiers are not publicly released;
the Talk Move → quality mapping below approximates their taxonomy.

Conversational Uptake (BERT, ddemszky/uptake-model) requires per-row speaker
labels (student vs. teacher), which Whisper alone does not provide. Uptake
scoring is therefore skipped unless the caller passes pre-diarized segments
with a 'speaker' field. Score is stored in AnnotatedUtterance.uptake_score.

Fallback: if edu-convokit is not installed, a keyword heuristic annotator is
used and a warning is emitted.
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass

from schemas import AnnotatedUtterance, Quality, TranscriptSegment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Talk Move label mapping (edu-convokit constants.py / YaHi/teacher_electra_small)
# ---------------------------------------------------------------------------

_TALK_MOVE_LABELS: dict[int, str] = {
    0: "No Move",
    1: "Keeping Everyone Together",
    2: "Relating to Another Student's Idea",
    3: "Restating",
    4: "Revoicing",
    5: "Pressing for Accuracy",
    6: "Pressing for Reasoning",
}

# Tutor CoPilot taxonomy approximation via Talk Move labels
# High-quality: student-centered, reasoning-eliciting moves
# High-pressure (flagged as risky per proposal §2): Pressing for Accuracy
_HIGH_QUALITY_MOVES: frozenset[str] = frozenset({
    "Keeping Everyone Together",
    "Relating to Another Student's Idea",
    "Restating",
    "Revoicing",
    "Pressing for Reasoning",
})
_HIGH_PRESSURE_MOVES: frozenset[str] = frozenset({"Pressing for Accuracy"})


def _talk_move_to_quality(label: str) -> Quality:
    if label in _HIGH_QUALITY_MOVES:
        return "high"
    if label in _HIGH_PRESSURE_MOVES:
        return "low"
    return "unknown"


# ---------------------------------------------------------------------------
# Direct HuggingFace ELECTRA annotator (YaHi/teacher_electra_small)
# Does not require edu-convokit — uses transformers pipeline directly.
# ---------------------------------------------------------------------------

class ElectraAnnotator:
    """
    Talk Moves classifier using YaHi/teacher_electra_small via HuggingFace
    transformers.  Falls back gracefully if the model cannot be loaded.
    """

    MODEL_ID = "YaHi/teacher_electra_small"

    def __init__(self) -> None:
        try:
            from transformers import pipeline as hf_pipeline
            self._pipe = hf_pipeline(
                "text-classification",
                model=self.MODEL_ID,
                tokenizer=self.MODEL_ID,
                top_k=1,
                truncation=True,
                max_length=512,
            )
            logger.info("Loaded ELECTRA Talk Moves model: %s", self.MODEL_ID)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot load {self.MODEL_ID}: {exc}. "
                "Ensure transformers is installed: pip install transformers"
            ) from exc

    def annotate(
        self,
        segments: list[TranscriptSegment],
    ) -> list[AnnotatedUtterance]:
        if not segments:
            return []

        texts = [s.text for s in segments]
        # Run in one batch for efficiency
        try:
            results = self._pipe(texts, batch_size=32)
        except Exception as exc:
            logger.warning("ELECTRA inference failed (%s); returning unknown quality", exc)
            results = [[{"label": "LABEL_0", "score": 1.0}]] * len(texts)

        out: list[AnnotatedUtterance] = []
        for seg, res in zip(segments, results):
            top = res[0] if isinstance(res, list) else res
            raw_label = top.get("label", "LABEL_0")
            # Labels may come back as "LABEL_0" .. "LABEL_6" or string names
            if raw_label.startswith("LABEL_"):
                idx = int(raw_label.split("_")[1])
                label = _TALK_MOVE_LABELS.get(idx, "No Move")
            else:
                label = raw_label
            quality = _talk_move_to_quality(label)
            out.append(
                AnnotatedUtterance(
                    t_start=seg.t_start,
                    t_end=seg.t_end,
                    text=seg.text,
                    talk_move=label,
                    strategy_quality=quality,
                    uptake_score=None,
                    notes=f"ELECTRA {self.MODEL_ID}",
                )
            )
        return out


# ---------------------------------------------------------------------------
# Edu-ConvoKit annotator (wraps edu-convokit if installed)
# ---------------------------------------------------------------------------

class EduConvoKitAnnotator:
    """
    Wraps edu-convokit's Annotator to classify teacher Talk Moves via ELECTRA.

    Usage::

        annotator = EduConvoKitAnnotator()
        utterances = annotator.annotate(segments)
    """

    def __init__(self) -> None:
        try:
            from edu_convokit.annotation.annotator import Annotator
            self._annotator = Annotator()
        except ImportError as exc:
            raise RuntimeError(
                "edu-convokit is not installed. "
                "Run: pip install edu-convokit"
            ) from exc

    def annotate(
        self,
        segments: list[TranscriptSegment],
    ) -> list[AnnotatedUtterance]:
        import pandas as pd  # lazy import: only needed with edu-convokit

        if not segments:
            return []

        df = pd.DataFrame(
            [{"t_start": s.t_start, "t_end": s.t_end, "text": s.text} for s in segments]
        )

        # Talk Moves classification (ELECTRA, 7-class)
        df = self._annotator.get_teacher_talk_moves(
            df,
            text_column="text",
            output_column="talk_move_idx",
        )

        out: list[AnnotatedUtterance] = []
        for _, row in df.iterrows():
            idx = row.get("talk_move_idx")
            if idx is None or pd.isna(idx):
                label = "No Move"
            else:
                label = _TALK_MOVE_LABELS.get(int(idx), "No Move")

            quality = _talk_move_to_quality(label)

            out.append(
                AnnotatedUtterance(
                    t_start=float(row["t_start"]),
                    t_end=float(row["t_end"]),
                    text=str(row["text"]),
                    talk_move=label,
                    strategy_quality=quality,
                    uptake_score=None,  # requires speaker diarization; see module docstring
                    notes="edu-convokit ELECTRA (YaHi/teacher_electra_small)",
                )
            )
        return out


# ---------------------------------------------------------------------------
# Heuristic fallback (kept for offline / no-GPU environments)
# ---------------------------------------------------------------------------

_LOW_QUALITY_PATTERNS = [
    r"\b(the answer is|here'?s the answer|it'?s \d+)\b",
    r"\b(let me (just )?(tell|show) you)\b",
    r"\byou (should |just )(use|plug|multiply|divide)\b",
    r"\bthis (always|never) works\b",
    r"\bthe formula is\b",
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
                    notes="heuristic fallback; install edu-convokit for ELECTRA classifier",
                )
            )
        return out


# ---------------------------------------------------------------------------
# Factory: pick the best available annotator
# ---------------------------------------------------------------------------

def make_annotator() -> ElectraAnnotator | EduConvoKitAnnotator | HeuristicStrategyAnnotator:
    """
    Return the best available annotator:
      1. ElectraAnnotator  — direct HuggingFace (YaHi/teacher_electra_small)
      2. EduConvoKitAnnotator — via edu-convokit wrapper
      3. HeuristicStrategyAnnotator — keyword fallback
    """
    try:
        ann = ElectraAnnotator()
        logger.info("Using direct ELECTRA Talk Moves annotator (YaHi/teacher_electra_small).")
        return ann
    except (RuntimeError, Exception) as exc:
        logger.debug("ELECTRA unavailable: %s", exc)

    try:
        ann = EduConvoKitAnnotator()
        logger.info("Using Edu-ConvoKit ELECTRA Talk Moves annotator.")
        return ann
    except (RuntimeError, Exception) as exc:
        logger.debug("edu-convokit unavailable: %s", exc)

    warnings.warn(
        "No ML annotator available. "
        "Install transformers for ELECTRA: pip install transformers torch. "
        "Falling back to keyword heuristic annotator.",
        stacklevel=2,
    )
    return HeuristicStrategyAnnotator()
