"""End-to-end session processing."""

from __future__ import annotations

from pathlib import Path

from config import PipelineConfig
from fusion import build_alignment_bins, heuristic_candidates, strategy_only_candidates
from language import make_annotator, transcribe_video
from report import save_report
from schemas import FrictionWindow, FusionResult, TeacherFrictionReport
from vision import (
    build_confusion_timeline,
    extract_frames_uniform_fps,
    get_video_duration_sec,
    sliding_window_average,
)
from vision.clip_video import clip_video_head
from vision.ocr_board import ocr_frames


def run_pipeline(video_path: str | Path, cfg: PipelineConfig | None = None) -> TeacherFrictionReport:
    cfg = cfg or PipelineConfig()
    video_path = Path(video_path).resolve()
    cfg.work_dir = Path(cfg.work_dir).resolve()
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = cfg.work_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    if cfg.max_duration_sec is not None and cfg.max_duration_sec > 0:
        clipped = cfg.work_dir / "_session_head.mp4"
        video_path = clip_video_head(video_path, clipped, float(cfg.max_duration_sec))

    duration = get_video_duration_sec(video_path)

    frames = extract_frames_uniform_fps(video_path, cfg.vision_sample_fps, frames_dir)
    frame_index = [(f.t_sec, f.path) for f in frames]

    raw_conf = build_confusion_timeline(frames, cfg.deepface_enforce_detection)
    smoothed = sliding_window_average(raw_conf, cfg.confusion_sliding_window_sec)

    segments = transcribe_video(
        video_path,
        model_size=cfg.whisper_model_size,
        device=cfg.whisper_device,
        compute_type=cfg.whisper_compute_type,
    )
    annotator = make_annotator()
    utterances = annotator.annotate(segments)

    if duration <= 0:
        duration = max((f.t_sec for f in frames), default=0.0)
        duration = max(duration, max((s.t_end for s in segments), default=0.0), 1.0)

    bins = build_alignment_bins(smoothed, utterances, cfg.alignment_grid_sec, duration)
    candidates, _mu, _sigma = heuristic_candidates(
        bins, cfg.confusion_z_threshold, frame_index,
        flag_unknown=cfg.flag_unknown_strategy,
    )
    # When DeepFace finds no faces the confusion signal is flat (all zeros),
    # so heuristic_candidates selects nothing. Fall back to strategy-only mode.
    if not candidates and all(b.mean_confusion == 0.0 for b in bins):
        candidates = strategy_only_candidates(
            bins, frame_index, flag_unknown=cfg.flag_unknown_strategy
        )
    candidates = candidates[: cfg.max_fusion_windows]

    fusion_results: list[FusionResult] = []
    claude = None

    if not cfg.skip_fusion and candidates:
        from fusion.claude_fusion import ClaudeFrictionFusion
        claude = ClaudeFrictionFusion(temperature=cfg.llm_temperature)

    for c in candidates:
        paths = c.frame_paths
        if len(paths) > cfg.frames_per_candidate and cfg.frames_per_candidate > 1:
            step = (len(paths) - 1) / (cfg.frames_per_candidate - 1)
            idxs = sorted(
                {min(len(paths) - 1, int(round(i * step))) for i in range(cfg.frames_per_candidate)}
            )
            paths = [paths[i] for i in idxs]
        elif len(paths) > cfg.frames_per_candidate:
            paths = paths[:1]

        confusion_txt = (
            f"mean confusion in window={c.mean_confusion:.3f}, z vs session={c.confusion_z:.2f}"
        )
        strat = c.strategy
        strategy_txt = (
            f"dominant_quality={strat.dominant_quality}, "
            f"low_quality={strat.low_quality}, high_pressure={strat.high_pressure}, "
            f"moves={strat.talk_moves}"
        )
        excerpt = strat.transcript_excerpt or "(no transcript overlap in bin)"

        board_text = ""
        if cfg.enable_ocr and paths:
            try:
                board_text = ocr_frames(paths, board_crop_ratio=cfg.ocr_board_crop_ratio)
            except Exception:
                pass  # OCR failure is non-fatal

        if claude is None:
            fusion_results.append(
                FusionResult(
                    t_start=c.t_start,
                    t_end=c.t_end,
                    friction=None,
                    rationale="fusion skipped",
                    alternative_strategy="",
                    raw_model_output=None,
                )
            )
            continue

        window = FrictionWindow(
            t_start_sec=c.t_start,
            t_end_sec=c.t_end,
            confusion_summary=confusion_txt,
            strategy_summary=strategy_txt,
            transcript_excerpt=excerpt,
            frame_paths=paths,
            board_text=board_text,
        )
        raw = claude.analyze_window(window)
        fusion_results.append(
            FusionResult(
                t_start=c.t_start,
                t_end=c.t_end,
                friction=raw.get("friction"),
                rationale=str(raw.get("rationale", "")),
                alternative_strategy=str(raw.get("alternative_strategy", "")),
                raw_model_output=raw,
            )
        )

    report = TeacherFrictionReport(
        video_path=str(video_path),
        bins=bins,
        candidates=candidates,
        fusion=fusion_results,
        work_dir=str(cfg.work_dir),
    )
    save_report(report, cfg.work_dir)
    return report
