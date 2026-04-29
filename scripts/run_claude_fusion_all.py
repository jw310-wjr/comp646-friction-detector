#!/usr/bin/env python3
"""
Batch Claude multimodal fusion for all 54 lessons.

For each risky bin identified by the heuristic transcript run, selects
up to 3 frames from the video pipeline and sends them to Claude along
with the strategy/transcript context.

Usage:
    export ANTHROPIC_API_KEY=...
    python scripts/run_claude_fusion_all.py
    python scripts/run_claude_fusion_all.py --workers 8 --max-bins-per-lesson 20
    python scripts/run_claude_fusion_all.py --lesson M-AU1  # single lesson
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fusion.claude_fusion import ClaudeFrictionFusion

TRANSCRIPT_RUNS  = ROOT / "runs" / "transcripts"
VIDEO_RUNS       = ROOT / "runs" / "video_pipeline"
FUSION_OUT       = ROOT / "runs" / "claude_fusion"
FRAMES_PER_BIN   = 3   # how many frames to send per bin


@dataclass
class RiskyBin:
    lesson_id: str
    t_start: float
    t_end: float
    dominant_quality: str
    low_quality: bool
    high_pressure: bool
    talk_moves: list
    transcript_excerpt: str
    frame_paths: list


def find_frames_for_bin(frames_dir: Path, t_start: float, t_end: float) -> list[Path]:
    """Pick up to FRAMES_PER_BIN frames evenly spaced within [t_start, t_end]."""
    if not frames_dir.exists():
        return []
    # frames named like f_000000_0.000s.jpg
    all_frames = sorted(frames_dir.glob("f_*.jpg"))
    # parse timestamps from filenames
    timed = []
    for f in all_frames:
        parts = f.stem.split("_")
        if len(parts) >= 3:
            try:
                t = float(parts[2].rstrip("s"))
                if t_start <= t <= t_end:
                    timed.append((t, f))
            except ValueError:
                pass
    if not timed:
        return []
    # pick evenly spaced subset
    if len(timed) <= FRAMES_PER_BIN:
        return [f for _, f in timed]
    step = len(timed) / FRAMES_PER_BIN
    indices = [int(i * step) for i in range(FRAMES_PER_BIN)]
    return [timed[i][1] for i in indices]


def load_risky_bins(lesson_id: str, max_bins: int | None) -> list[RiskyBin]:
    summary_path = TRANSCRIPT_RUNS / lesson_id / "summary.json"
    if not summary_path.exists():
        return []
    data = json.loads(summary_path.read_text())
    frames_dir = VIDEO_RUNS / lesson_id / "frames"

    bins = []
    risky = data.get("risky_bins", [])
    if max_bins:
        risky = risky[:max_bins]

    for b in risky:
        fps = find_frames_for_bin(frames_dir, b["t_start"], b["t_end"])
        bins.append(RiskyBin(
            lesson_id=lesson_id,
            t_start=b["t_start"],
            t_end=b["t_end"],
            dominant_quality=b["dominant_quality"],
            low_quality=b.get("low_quality", False),
            high_pressure=b.get("high_pressure", False),
            talk_moves=b.get("talk_moves", []),
            transcript_excerpt=b.get("excerpt", ""),
            frame_paths=[str(f) for f in fps],
        ))
    return bins


def process_bin(fusion: ClaudeFrictionFusion, rb: RiskyBin) -> dict:
    """Call Claude on one bin; returns result dict."""

    class _FakeBin:
        t_start = rb.t_start
        t_end = rb.t_end
        mean_confusion = 0.0
        confusion_z = 0.0
        frame_paths = rb.frame_paths

        class strategy:
            dominant_quality = rb.dominant_quality
            low_quality = rb.low_quality
            high_pressure = rb.high_pressure
            talk_moves = rb.talk_moves
            transcript_excerpt = rb.transcript_excerpt

    results = fusion.analyze_candidates([_FakeBin()])
    r = results[0]
    r["n_frames_sent"] = len(rb.frame_paths)
    return r


def run_lesson(lesson_id: str, bins: list[RiskyBin], out_dir: Path) -> dict:
    out_path = out_dir / "fusion_heuristic.json"
    if out_path.exists():
        # resume: load existing and skip already-done bins
        existing = json.loads(out_path.read_text())
        done_keys = {(r["t_start"], r["t_end"]) for r in existing}
        bins = [b for b in bins if (b.t_start, b.t_end) not in done_keys]
        results = existing
    else:
        results = []

    if not bins:
        friction_yes = sum(1 for r in results if r.get("friction") is True)
        return {
            "lesson_id": lesson_id,
            "total": len(results),
            "friction_yes": friction_yes,
            "friction_rate": round(100 * friction_yes / len(results), 1) if results else 0,
        }

    fusion = ClaudeFrictionFusion()
    friction_yes = sum(1 for r in results if r.get("friction") is True)

    for b in bins:
        try:
            r = process_bin(fusion, b)
            results.append(r)
            if r.get("friction"):
                friction_yes += 1
        except Exception as e:
            results.append({
                "t_start": b.t_start, "t_end": b.t_end,
                "friction": None, "rationale": f"ERROR: {e}",
                "alternative_strategy": "", "n_frames_sent": 0,
            })
        # save after each bin for resumability
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    return {
        "lesson_id": lesson_id,
        "total": len(results),
        "friction_yes": friction_yes,
        "friction_rate": round(100 * friction_yes / len(results), 1) if results else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lesson", default=None, help="Run single lesson (e.g. M-AU1)")
    ap.add_argument("--workers", type=int, default=4, help="Parallel lessons (default 4)")
    ap.add_argument("--max-bins-per-lesson", type=int, default=None,
                    help="Cap risky bins per lesson (default: all)")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: ANTHROPIC_API_KEY not set")

    # collect lessons
    if args.lesson:
        lesson_ids = [args.lesson]
    else:
        lesson_ids = sorted(d.name for d in TRANSCRIPT_RUNS.iterdir()
                            if d.is_dir() and (d / "summary.json").exists())

    print(f"Lessons to process: {len(lesson_ids)}")

    # load bins
    jobs = []
    total_bins = 0
    for lid in lesson_ids:
        bins = load_risky_bins(lid, args.max_bins_per_lesson)
        out_dir = FUSION_OUT / lid
        out_dir.mkdir(parents=True, exist_ok=True)
        jobs.append((lid, bins, out_dir))
        total_bins += len(bins)
    print(f"Total risky bins to process: {total_bins}")
    print(f"Workers: {args.workers}\n")

    summary_rows = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_lesson, lid, bins, out_dir): lid
                   for lid, bins, out_dir in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            lid = futures[fut]
            try:
                result = fut.result()
                rate = result.get("friction_rate", "?")
                skipped = result.get("skipped", False)
                status = "already done" if skipped else f"friction_rate={rate}%"
                elapsed = time.time() - t0
                print(f"[{i:02d}/{len(jobs)}] {lid:10s}  {status}  ({elapsed:.0f}s elapsed)")
                summary_rows.append(result)
            except Exception as e:
                print(f"[{i:02d}/{len(jobs)}] {lid:10s}  ERROR: {e}")

    # Save aggregate summary
    agg_path = FUSION_OUT / "aggregate_fusion.json"
    agg_path.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False))

    total_confirmed = sum(r.get("friction_yes", 0) for r in summary_rows)
    total_processed = sum(r.get("total", 0) for r in summary_rows)
    overall_rate = round(100 * total_confirmed / total_processed, 1) if total_processed else 0

    print(f"\n=== DONE in {time.time()-t0:.0f}s ===")
    print(f"Total bins analyzed : {total_processed}")
    print(f"Friction confirmed  : {total_confirmed} ({overall_rate}%)")
    print(f"Results saved to    : {FUSION_OUT}")


if __name__ == "__main__":
    main()
