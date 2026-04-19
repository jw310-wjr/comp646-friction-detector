#!/usr/bin/env python3
"""
Run Claude API fusion on heuristic-flagged risky bins for a given lesson.

Reads the summary.json produced by run_transcript.py (heuristic annotator),
sends each risky bin to ClaudeFrictionFusion, and saves results to
runs/claude_fusion/<lesson_id>/fusion_results.json

Usage:
    python scripts/run_claude_fusion.py --lesson M-AU1
    python scripts/run_claude_fusion.py --lesson M-AU1 --annotator electra
    python scripts/run_claude_fusion.py --lesson M-AU1 --all-bins   # run on all bins, not just risky
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fusion.claude_fusion import ClaudeFrictionFusion


@dataclass
class SimpleBin:
    t_start: float
    t_end: float
    mean_confusion: float
    confusion_z: float
    frame_paths: list
    strategy: "SimpleBinStrategy"


@dataclass
class SimpleBinStrategy:
    dominant_quality: str
    low_quality: bool
    high_pressure: bool
    talk_moves: list
    transcript_excerpt: str


def load_bins(summary_path: Path, risky_only: bool = True) -> list[SimpleBin]:
    data = json.loads(summary_path.read_text())
    bins = []
    for b in data["bins"]:
        strat = SimpleBinStrategy(
            dominant_quality=b["dominant_quality"],
            low_quality=b.get("low_quality", False),
            high_pressure=b.get("high_pressure", False),
            talk_moves=b.get("talk_moves", []),
            transcript_excerpt=b.get("excerpt", ""),
        )
        is_risky = b.get("low_quality") or b.get("high_pressure") or b["dominant_quality"] in ("low", "unknown")
        if risky_only and not is_risky:
            continue
        bins.append(SimpleBin(
            t_start=b["t_start"],
            t_end=b["t_end"],
            mean_confusion=0.0,
            confusion_z=0.0,
            frame_paths=[],
            strategy=strat,
        ))
    return bins


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lesson", default="M-AU1")
    ap.add_argument("--annotator", choices=["heuristic", "electra"], default="heuristic")
    ap.add_argument("--all-bins", action="store_true", help="Run on all bins, not just risky")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    args = ap.parse_args()

    run_dir = "transcripts" if args.annotator == "heuristic" else "transcripts_electra"
    summary_path = ROOT / "runs" / run_dir / args.lesson / "summary.json"
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found. Run run_transcript.py first.")
        sys.exit(1)

    bins = load_bins(summary_path, risky_only=not args.all_bins)
    print(f"Lesson: {args.lesson} | Annotator: {args.annotator} | Bins to process: {len(bins)}")

    out_dir = ROOT / "runs" / "claude_fusion" / args.lesson
    out_dir.mkdir(parents=True, exist_ok=True)

    fusion = ClaudeFrictionFusion(model=args.model)
    results = []

    for i, b in enumerate(bins):
        m_s, s_s = int(b.t_start)//60, int(b.t_start)%60
        m_e, s_e = int(b.t_end)//60, int(b.t_end)%60
        print(f"  [{i+1}/{len(bins)}] {m_s}:{s_s:02d}–{m_e}:{s_e:02d} "
              f"quality={b.strategy.dominant_quality} moves={b.strategy.talk_moves}", end=" ... ", flush=True)
        result = fusion.analyze_candidates([b])
        r = result[0]
        print(f"friction={r.get('friction')}")
        results.append(r)

    out_path = out_dir / f"fusion_{args.annotator}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n✓ Saved {len(results)} results → {out_path}")

    # Summary stats
    friction_yes = sum(1 for r in results if r.get("friction") is True)
    friction_no  = sum(1 for r in results if r.get("friction") is False)
    friction_unk = len(results) - friction_yes - friction_no
    print(f"\nFusion summary:")
    print(f"  friction=True : {friction_yes}")
    print(f"  friction=False: {friction_no}")
    print(f"  uncertain     : {friction_unk}")


if __name__ == "__main__":
    main()
