"""Trim session video to first N seconds (copies streams; keeps audio for Whisper)."""

from __future__ import annotations

import subprocess
from pathlib import Path


def clip_video_head(src: str | Path, dst: str | Path, max_duration_sec: float) -> Path:
    """Write ``dst`` = first ``max_duration_sec`` of ``src`` using bundled ffmpeg."""
    import imageio_ffmpeg as ioff

    src, dst = Path(src).resolve(), Path(dst).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    ff = ioff.get_ffmpeg_exe()
    cmd = [
        ff,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-t",
        str(max_duration_sec),
        "-c",
        "copy",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    return dst
