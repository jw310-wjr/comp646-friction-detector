"""
Multimodal friction reasoning via Anthropic Claude API.

Replaces Qwen2.5-VL for environments without a GPU.  When frame images are
available they are base64-encoded and sent as vision content; in
transcript-only mode the same prompt is sent as pure text.

Usage:
    from fusion.claude_fusion import ClaudeFrictionFusion
    fusion = ClaudeFrictionFusion()          # uses ANTHROPIC_API_KEY env var
    result = fusion.analyze_window(window)   # returns dict with friction/rationale/alternative_strategy
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import anthropic

from schemas import FrictionWindow

DEFAULT_MODEL = "claude-sonnet-4-6"


def _encode_image(path: str | Path) -> tuple[str, str]:
    """Return (base64_data, media_type) for a local image file."""
    p = Path(path)
    suffix = p.suffix.lower()
    media = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
             ".png": "image/png", ".gif": "image/gif",
             ".webp": "image/webp"}.get(suffix, "image/jpeg")
    data = base64.standard_b64encode(p.read_bytes()).decode("utf-8")
    return data, media


def _instruction_text(w: FrictionWindow) -> str:
    board_section = (
        f"\nBoard / slide content (OCR):\n\"\"\"{w.board_text}\"\"\"\n"
        if w.board_text.strip()
        else ""
    )
    return (
        f"You are an education researcher helping a teacher reflect on a classroom clip.\n\n"
        f"Time window: {w.t_start_sec:.1f}s – {w.t_end_sec:.1f}s\n\n"
        f"Confusion signal (from vision / timeline): {w.confusion_summary}\n\n"
        f"Instructional strategy signal (from dialogue classifiers): {w.strategy_summary}\n\n"
        f"Nearby transcript (excerpt):\n\"\"\"{w.transcript_excerpt}\"\"\""
        f"{board_section}\n"
        f"Task:\n"
        f"1) Decide whether this is a genuine \"pedagogical friction\" moment "
        f"(low-quality or high-pressure strategy coinciding with student confusion). "
        f"Answer yes/no with one sentence of justification.\n"
        f"2) If yes, suggest ONE alternative teacher move (e.g., ask a guiding question, "
        f"press for reasoning, revoice a student idea). Be specific to this excerpt.\n\n"
        f"Respond as compact JSON with keys: "
        f"friction (bool), rationale (str), alternative_strategy (str). No markdown."
    )


class ClaudeFrictionFusion:
    """Claude-backed drop-in replacement for QwenVLFrictionFusion."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 512,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    def _build_content(self, w: FrictionWindow) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []

        # Attach up to 4 frames if available
        for fp in w.frame_paths[:4]:
            try:
                data, media_type = _encode_image(fp)
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": data},
                })
            except Exception:
                pass  # skip unreadable frames silently

        content.append({"type": "text", "text": _instruction_text(w)})
        return content

    def analyze_window(self, w: FrictionWindow) -> dict[str, Any]:
        content = self._build_content(w)
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system="You output only valid JSON matching the user schema.",
            messages=[{"role": "user", "content": content}],
        )
        raw = response.content[0].text
        return _parse_json(raw)

    def analyze_candidates(
        self, candidates: list[Any]
    ) -> list[dict[str, Any]]:
        """Run fusion on a list of FrictionCandidate objects."""
        results = []
        for c in candidates:
            w = FrictionWindow(
                t_start_sec=c.t_start,
                t_end_sec=c.t_end,
                confusion_summary=(
                    f"mean confusion={c.mean_confusion:.2f}, z={c.confusion_z:.2f}"
                    if c.mean_confusion > 0
                    else "transcript-only mode (no vision signal)"
                ),
                strategy_summary=(
                    f"dominant_quality={c.strategy.dominant_quality}, "
                    f"talk_moves={c.strategy.talk_moves}"
                ),
                transcript_excerpt=c.strategy.transcript_excerpt,
                frame_paths=c.frame_paths,
            )
            result = self.analyze_window(w)
            result["t_start"] = c.t_start
            result["t_end"] = c.t_end
            results.append(result)
        return results


def _parse_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start: end + 1])
            except json.JSONDecodeError:
                pass
        return {"friction": None, "rationale": raw, "alternative_strategy": ""}


def load_fusion(model: str = DEFAULT_MODEL, **kwargs: Any) -> ClaudeFrictionFusion:
    """Convenience constructor matching the Qwen module's interface."""
    return ClaudeFrictionFusion(model=model, **kwargs)
