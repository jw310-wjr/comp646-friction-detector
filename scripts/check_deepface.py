#!/usr/bin/env python3
"""
Sanity-check DeepFace confusion signals across all run outputs.

Reads teacher_friction_report.json files in runs/video_pipeline/ and
reports per-lesson confusion statistics.  All-zero lessons indicate that
DeepFace found no usable faces (wide-angle camera, face too small, etc.).

Usage:
    python scripts/check_deepface.py
    python scripts/check_deepface.py --runs-dir /scratch/jw310/comp646/runs/video_pipeline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ROOT / "runs" / "video_pipeline"


def check(runs_dir: Path) -> None:
    reports = sorted(runs_dir.glob("*/teacher_friction_report.json"))
    if not reports:
        print(f"No reports found in {runs_dir}")
        return

    all_zero, nonzero = [], []
    for rp in reports:
        lesson = rp.parent.name
        try:
            data = json.loads(rp.read_text())
        except Exception as e:
            print(f"  [WARN] {lesson}: could not read report ({e})")
            continue

        bins = data.get("bins", [])
        scores = [b.get("mean_confusion", 0.0) for b in bins]
        max_c = max(scores, default=0.0)
        n_bins = len(bins)

        if max_c == 0.0:
            all_zero.append((lesson, n_bins))
        else:
            nonzero.append((lesson, n_bins, max_c))

    print(f"{'='*60}")
    print(f"DeepFace sanity check  ({len(reports)} lessons)")
    print(f"{'='*60}")
    print(f"\nLessons with non-zero confusion ({len(nonzero)}):")
    for lesson, n, mx in sorted(nonzero, key=lambda x: -x[2]):
        print(f"  {lesson:<12}  bins={n:>3}  max_confusion={mx:.3f}")

    print(f"\nAll-zero lessons ({len(all_zero)}) — DeepFace found no usable faces:")
    for lesson, n in sorted(all_zero):
        print(f"  {lesson:<12}  bins={n:>3}")

    if all_zero:
        print(
            f"\n[NOTE] {len(all_zero)}/{len(reports)} lessons have zero confusion signal.\n"
            f"       Likely cause: wide-angle classroom cameras render faces <20 px.\n"
            f"       Pipeline falls back to strategy-only candidate selection for these."
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    args = ap.parse_args()
    check(args.runs_dir)


if __name__ == "__main__":
    main()
