"""Pipeline hyperparameters aligned with the COMP 646 proposal."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    vision_sample_fps: float = 2.0
    confusion_sliding_window_sec: float = 30.0
    alignment_grid_sec: float = 30.0
    confusion_z_threshold: float = 1.0
    whisper_model_size: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    qwen_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    qwen_max_new_tokens: int = 512
    work_dir: Path = field(default_factory=lambda: Path("./friction_work"))
    skip_fusion: bool = False
    max_fusion_windows: int = 20
    frames_per_candidate: int = 4
    deepface_enforce_detection: bool = False
    # If set, process only the first N seconds (full lessons are very slow on CPU).
    max_duration_sec: float | None = None
    # Treat bins with 'unknown' strategy quality as risky (recommended when using heuristics).
    flag_unknown_strategy: bool = True
