"""
Parse TIMSS-style .txt transcripts into TranscriptSegment lists.

Expected line format:
    HH:MM:SS\tSpeaker\tText

Speaker codes:
    T   = Teacher
    S, SN, S1, S2, ...  = Students

Returns all utterances as TranscriptSegment objects.  Speaker identity is
preserved in the segment text as "[T]" or "[S]" prefix so downstream strategy
annotators can optionally filter to teacher turns only.
"""

from __future__ import annotations

import re
from pathlib import Path

from schemas import TranscriptSegment


def _ts_to_sec(ts: str) -> float:
    """Convert HH:MM:SS (or MM:SS) to seconds."""
    parts = ts.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(parts[0])


_TS_RE = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?$")


def parse_transcript(
    path: str | Path,
    teacher_only: bool = False,
    default_utterance_duration: float = 10.0,
) -> list[TranscriptSegment]:
    """
    Parse a TIMSS transcript .txt file.

    Parameters
    ----------
    path:
        Path to the transcript file.
    teacher_only:
        If True, return only teacher (T) utterances.
    default_utterance_duration:
        Duration (seconds) assigned to the final utterance (no next-line
        timestamp available to infer from).

    Returns
    -------
    List of TranscriptSegment sorted by t_start.
    """
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()

    records: list[tuple[float, str, str]] = []  # (t_start, speaker, text)

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 2)
        if len(parts) < 2:
            continue
        ts_raw, speaker = parts[0].strip(), parts[1].strip()
        text = parts[2].strip() if len(parts) == 3 else ""

        if not _TS_RE.match(ts_raw):
            continue  # skip header or malformed rows

        t_start = _ts_to_sec(ts_raw)
        records.append((t_start, speaker, text))

    if not records:
        return []

    # Infer t_end from next utterance's t_start
    segments: list[TranscriptSegment] = []
    for i, (t_start, speaker, text) in enumerate(records):
        if i + 1 < len(records):
            t_end = records[i + 1][0]
            # Clamp: each utterance is at least 1 second, at most 120 s
            t_end = max(t_start + 1.0, t_end)
            t_end = min(t_start + 120.0, t_end)
        else:
            t_end = t_start + default_utterance_duration

        if teacher_only and speaker.upper() != "T":
            continue

        label = "[T]" if speaker.upper() == "T" else "[S]"
        segments.append(
            TranscriptSegment(
                t_start=t_start,
                t_end=t_end,
                text=f"{label} {text}" if text else label,
            )
        )

    return segments
