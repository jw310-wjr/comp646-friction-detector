#!/usr/bin/env python3
"""
Run the language-only friction pipeline on TIMSS transcript .txt files.

Usage
-----
# All transcripts in catalog.csv:
python scripts/run_transcript.py

# One specific transcript:
python scripts/run_transcript.py --lesson M-AU1

# Choose output root:
python scripts/run_transcript.py --out-dir ./runs/transcripts
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA = ROOT / "data"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fusion import build_alignment_bins, strategy_only_candidates
from language import make_annotator, parse_transcript
from report import save_report
from schemas import FusionResult, TeacherFrictionReport


GRID_SEC = 30.0  # bin width in seconds

# flag_unknown=True is recommended only for the keyword heuristic annotator.
# With ELECTRA, "No Move" (→ unknown quality) is a valid classification meaning
# the teacher didn't use a specific talk move — not necessarily friction.
# We detect the annotator type at runtime and set accordingly.
_HEURISTIC_ANNOTATOR_CLS = "HeuristicStrategyAnnotator"


def run_one(
    lesson_id: str,
    transcript_path: Path,
    out_dir: Path,
    annotator=None,
) -> TeacherFrictionReport:
    """Analyse one transcript and return (and save) the friction report."""
    if annotator is None:
        annotator = make_annotator()

    # 1. Parse transcript ------------------------------------------------
    # Use teacher-only utterances for strategy annotation: student short turns
    # (e.g. "Okay." or "(inaudible)") would otherwise inflate "unknown" bins.
    teacher_segs = parse_transcript(transcript_path, teacher_only=True)
    if not teacher_segs:
        warnings.warn(f"[{lesson_id}] No teacher segments parsed from {transcript_path}")

    # Duration from all utterances (teacher + student)
    all_segs = parse_transcript(transcript_path, teacher_only=False)
    duration = max((s.t_end for s in all_segs), default=GRID_SEC)

    # 2. Strategy annotation (teacher turns only) -----------------------
    utterances = annotator.annotate(teacher_segs)

    # 3. Build alignment bins (no vision → confusion = 0 everywhere) ---
    bins = build_alignment_bins(
        smoothed_confusion=[],   # empty → all mean_confusion = 0.0
        utterances=utterances,
        grid_sec=GRID_SEC,
        video_duration_sec=duration,
    )

    # 4. Strategy-only candidate selection ------------------------------
    # flag_unknown only for heuristic annotator; ELECTRA "No Move" ≠ friction.
    flag_unknown = type(annotator).__name__ == _HEURISTIC_ANNOTATOR_CLS
    candidates = strategy_only_candidates(
        bins,
        frame_index=[],          # no frames available
        flag_unknown=flag_unknown,
    )

    # 5. No VLM fusion in transcript-only mode -------------------------
    fusion_results: list[FusionResult] = [
        FusionResult(
            t_start=c.t_start,
            t_end=c.t_end,
            friction=None,
            rationale="transcript-only mode — no vision/VLM fusion",
            alternative_strategy="",
            raw_model_output=None,
        )
        for c in candidates
    ]

    report = TeacherFrictionReport(
        video_path=str(transcript_path),   # repurpose field for transcript path
        bins=bins,
        candidates=candidates,
        fusion=fusion_results,
        work_dir=str(out_dir),
    )
    save_report(report, out_dir)

    # Also dump a compact per-lesson summary JSON for easy downstream use
    summary = {
        "lesson_id": lesson_id,
        "transcript": str(transcript_path),
        "n_bins": len(bins),
        "n_risky_bins": len(candidates),
        "bins": [
            {
                "t_start": b.t_start,
                "t_end": b.t_end,
                "dominant_quality": b.strategy.dominant_quality,
                "low_quality": b.strategy.low_quality,
                "high_pressure": b.strategy.high_pressure,
                "talk_moves": b.strategy.talk_moves,
                "excerpt": b.strategy.transcript_excerpt[:300],
            }
            for b in bins
        ],
        "risky_bins": [
            {
                "t_start": c.t_start,
                "t_end": c.t_end,
                "dominant_quality": c.strategy.dominant_quality,
                "low_quality": c.strategy.low_quality,
                "high_pressure": c.strategy.high_pressure,
                "talk_moves": c.strategy.talk_moves,
                "excerpt": c.strategy.transcript_excerpt[:500],
            }
            for c in candidates
        ],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Language-only friction analysis on TIMSS transcript files"
    )
    ap.add_argument(
        "--lesson",
        type=str,
        default=None,
        help="Lesson ID to process (e.g. M-AU1). Defaults to all lessons in catalog.csv.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "runs" / "transcripts",
        help="Root output directory. Each lesson gets its own subdirectory.",
    )
    ap.add_argument(
        "--catalog",
        type=Path,
        default=DATA / "catalog.csv",
        help="Path to catalog.csv",
    )
    args = ap.parse_args()

    catalog_path = args.catalog
    if not catalog_path.exists():
        sys.exit(f"Catalog not found: {catalog_path}")

    with catalog_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.lesson:
        rows = [r for r in rows if r["lesson_id"] == args.lesson]
        if not rows:
            sys.exit(f"Lesson '{args.lesson}' not found in catalog.")

    # Build annotator once (model load is expensive)
    print("Loading strategy annotator…")
    annotator = make_annotator()
    print(f"Annotator ready. Processing {len(rows)} lesson(s).\n")

    results_summary: list[dict] = []

    for row in rows:
        lesson_id = row["lesson_id"]
        tf = row.get("transcript_file", "").strip()
        if not tf:
            print(f"[{lesson_id}] No transcript file listed — skipping.")
            continue

        transcript_path = DATA / tf
        if not transcript_path.exists():
            print(f"[{lesson_id}] Transcript not found: {transcript_path} — skipping.")
            continue

        out_dir = args.out_dir / lesson_id
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{lesson_id}] Processing {transcript_path.name} …", end=" ", flush=True)
        try:
            report = run_one(lesson_id, transcript_path, out_dir, annotator=annotator)
            n_risky = len(report.candidates)
            n_bins = len(report.bins)
            print(f"{n_risky}/{n_bins} bins flagged as risky.")
            results_summary.append(
                {
                    "lesson_id": lesson_id,
                    "subject": row.get("subject", ""),
                    "country": row.get("country", ""),
                    "n_bins": n_bins,
                    "n_risky_bins": n_risky,
                    "pct_risky": round(100 * n_risky / n_bins, 1) if n_bins else 0,
                }
            )
        except Exception as exc:
            print(f"ERROR: {exc}")

    # Write aggregate results table
    if results_summary:
        out_agg = args.out_dir / "aggregate_results.json"
        out_agg.parent.mkdir(parents=True, exist_ok=True)
        out_agg.write_text(
            json.dumps(results_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nAggregate results → {out_agg}")

        # Print a simple text table
        print(
            f"\n{'Lesson':<12} {'Subj':<8} {'Country':<8} {'Bins':>5} {'Risky':>6} {'%Risky':>7}"
        )
        print("-" * 52)
        for r in results_summary:
            print(
                f"{r['lesson_id']:<12} {r['subject']:<8} {r['country']:<8} "
                f"{r['n_bins']:>5} {r['n_risky_bins']:>6} {r['pct_risky']:>6.1f}%"
            )


if __name__ == "__main__":
    main()
