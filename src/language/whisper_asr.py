"""Whisper transcription (proposal §2) via faster-whisper for CPU-friendly inference."""

from __future__ import annotations

from pathlib import Path

from schemas import TranscriptSegment


def transcribe_video(
    video_path: str | Path,
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
) -> list[TranscriptSegment]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, _info = model.transcribe(str(video_path), vad_filter=True)
    out: list[TranscriptSegment] = []
    for seg in segments:
        out.append(
            TranscriptSegment(
                t_start=float(seg.start),
                t_end=float(seg.end),
                text=seg.text.strip(),
            )
        )
    return out
