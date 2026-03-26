#!/usr/bin/env python3
"""Run full multimodal friction pipeline on one session video."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import PipelineConfig
from pipeline.run import run_pipeline


def main() -> None:
    p = argparse.ArgumentParser(description="COMP 646 — pedagogical friction detector")
    p.add_argument("--video", type=Path, required=True, help="Path to session video (video+audio)")
    p.add_argument("--work-dir", type=Path, default=Path("./friction_work"))
    p.add_argument("--vision-fps", type=float, default=2.0)
    p.add_argument("--grid-sec", type=float, default=30.0)
    p.add_argument("--z", type=float, default=1.0, help="Confusion excess threshold in σ units")
    p.add_argument("--whisper", type=str, default="base", help="faster-whisper model size")
    p.add_argument("--whisper-device", type=str, default="cpu")
    p.add_argument("--qwen-model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--skip-fusion", action="store_true", help="Run vision+language+heuristics only")
    p.add_argument("--max-candidates", type=int, default=20)
    p.add_argument(
        "--max-duration-sec",
        type=float,
        default=None,
        help="Process only the first N seconds (recommended on CPU for long TIMSS lessons)",
    )
    args = p.parse_args()

    cfg = PipelineConfig(
        work_dir=args.work_dir,
        vision_sample_fps=args.vision_fps,
        alignment_grid_sec=args.grid_sec,
        confusion_z_threshold=args.z,
        whisper_model_size=args.whisper,
        whisper_device=args.whisper_device,
        qwen_model_id=args.qwen_model,
        skip_fusion=args.skip_fusion,
        max_fusion_windows=args.max_candidates,
        max_duration_sec=args.max_duration_sec,
    )
    report = run_pipeline(args.video, cfg)
    out = Path(report.work_dir) / "teacher_friction_report.txt"
    print(f"Wrote report under: {report.work_dir}")
    print(out.read_text(encoding="utf-8")[:2000])


if __name__ == "__main__":
    main()
