"""End-to-end session processing."""

from __future__ import annotations

from pathlib import Path

from config import PipelineConfig
from fusion import build_alignment_bins, heuristic_candidates
from language import HeuristicStrategyAnnotator, transcribe_video
from report import save_report
from schemas import FrictionWindow, FusionResult, TeacherFrictionReport
from vision import (
    build_confusion_timeline,
    extract_frames_uniform_fps,
    get_video_duration_sec,
    sliding_window_average,
)
from vision.clip_video import clip_video_head


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
    annotator = HeuristicStrategyAnnotator()
    utterances = annotator.annotate(segments)

    if duration <= 0:
        duration = max((f.t_sec for f in frames), default=0.0)
        duration = max(duration, max((s.t_end for s in segments), default=0.0), 1.0)

    bins = build_alignment_bins(smoothed, utterances, cfg.alignment_grid_sec, duration)
    candidates, _mu, _sigma = heuristic_candidates(
        bins, cfg.confusion_z_threshold, frame_index,
        flag_unknown=cfg.flag_unknown_strategy,
    )
    candidates = candidates[: cfg.max_fusion_windows]

    fusion_results: list[FusionResult] = []
    qwen: QwenVLFrictionFusion | None = None

    if not cfg.skip_fusion and candidates:
        from fusion.qwen_multimodal import QwenVLFrictionFusion

        qwen = QwenVLFrictionFusion(
            model_id=cfg.qwen_model_id,
            max_new_tokens=cfg.qwen_max_new_tokens,
        )

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

        if qwen is None:
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
        )
        raw = qwen.analyze_window(window)
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
