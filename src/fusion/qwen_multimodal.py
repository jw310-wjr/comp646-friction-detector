"""
Multimodal friction reasoning with Qwen2.5-VL (replaces GPT-4o in the proposal).

Model card: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from schemas import FrictionWindow


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def _to_file_uri(path: str | Path) -> str:
    p = Path(path).resolve()
    return p.as_uri()


class QwenVLFrictionFusion:
    """
    Loads Qwen2.5-VL and runs instruction-tuned multimodal inference on
    (video frame(s) + dialogue + timeline cues) to confirm friction and suggest alternatives.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        max_new_tokens: int = 512,
        **model_kwargs: Any,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        dtype = model_kwargs.pop("torch_dtype", None) or "auto"
        attn = model_kwargs.pop("attn_implementation", None)
        load_kw: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": "auto",
            **model_kwargs,
        }
        if attn:
            load_kw["attn_implementation"] = attn
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            **load_kw,
        )
        self._processor = AutoProcessor.from_pretrained(model_id)

    def _build_user_content(self, w: FrictionWindow) -> list[dict[str, Any]]:
        frames = [_to_file_uri(f) for f in w.frame_paths]
        image_parts: list[dict[str, Any]] = [
            {"type": "image", "image": uri} for uri in frames
        ]
        text = self._instruction_text(w)
        return [*image_parts, {"type": "text", "text": text}]

    @staticmethod
    def _instruction_text(w: FrictionWindow) -> str:
        return f"""You are an education researcher helping a teacher reflect on a tutoring/classroom clip.

Time window: {w.t_start_sec:.1f}s – {w.t_end_sec:.1f}s

Confusion signal (from vision / timeline): {w.confusion_summary}

Instructional strategy signal (from dialogue classifiers): {w.strategy_summary}

Nearby transcript (excerpt):
\"\"\"{w.transcript_excerpt}\"\"\"

Task:
1) Decide whether this is a genuine "pedagogical friction" moment (low-quality or high-pressure strategy coinciding with clear student confusion). Answer yes/no briefly with one sentence of justification.
2) If yes, suggest ONE alternative teacher move from high-quality strategies (e.g., ask a guiding question, prompt explanation, press for reasoning, revoice/build on student idea). Be specific to this excerpt.

Respond as compact JSON with keys: friction (bool), rationale (str), alternative_strategy (str). No markdown."""

    def analyze_window(self, w: FrictionWindow) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": "You output only valid JSON matching the user schema.",
            },
            {
                "role": "user",
                "content": self._build_user_content(w),
            },
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        device = next(self._model.parameters()).device
        inputs = inputs.to(device)

        generated_ids = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
        )
        trimmed = [
            out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        raw = self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return self._parse_json_response(raw)

    @staticmethod
    def _parse_json_response(raw: str) -> dict[str, Any]:
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start >= 0 and end > start:
                return json.loads(raw[start : end + 1])
            return {"friction": None, "rationale": raw, "alternative_strategy": ""}


def load_fusion(model_id: str = DEFAULT_MODEL_ID, **kwargs: Any) -> QwenVLFrictionFusion:
    """Convenience constructor for pipeline wiring."""
    return QwenVLFrictionFusion(model_id=model_id, **kwargs)
