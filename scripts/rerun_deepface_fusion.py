#!/usr/bin/env python3
"""
Re-run DeepFace confusion scoring on already-extracted frames, then
rebuild alignment bins and run local Claude fusion.

Frames and Whisper transcripts are already in runs/video_pipeline/<lesson_id>/.
This script skips video I/O and only does the missing vision analysis.

Usage:
    python scripts/rerun_deepface_fusion.py --lesson M-HK1
    python scripts/rerun_deepface_fusion.py --all
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import statistics
from pathlib import Path
from collections import defaultdict

ROOT   = Path(__file__).resolve().parents[1]
SRC    = ROOT / "src"
VP_DIR = ROOT / "runs" / "video_pipeline"
sys.path.insert(0, str(SRC))

from vision.confusion_timeline import build_confusion_timeline, _EMOTION_CONFUSION_WEIGHT
from vision import sliding_window_average
from schemas import FrameSample


# ── Load existing frames ──────────────────────────────────────────────────────

def load_frames(frames_dir: Path) -> list[FrameSample]:
    """Reconstruct FrameSample list from saved JPEGs (filename encodes timestamp)."""
    samples = []
    for p in sorted(frames_dir.glob("f_*.jpg")):
        # filename: f_000000_0.000s.jpg
        m = re.search(r"_(\d+\.\d+)s\.jpg$", p.name)
        if m:
            samples.append(FrameSample(t_sec=float(m.group(1)), path=str(p.resolve())))
    return samples


# ── Build bins from existing report + new confusion scores ────────────────────

def rebuild_bins(report_json: dict, smoothed: list, grid_sec: float = 30.0) -> list[dict]:
    """
    Merge new confusion scores into the existing bins (which already have
    Whisper transcript + strategy annotation from the previous pipeline run).
    """
    conf_by_t = {pt.t_sec: pt.score for pt in smoothed}

    new_bins = []
    for b in report_json["bins"]:
        t_s, t_e = b["t_start"], b["t_end"]
        # Average smoothed confusion over the bin window
        window_scores = [
            v for t, v in conf_by_t.items()
            if t_s <= t < t_e
        ]
        mean_conf = float(statistics.mean(window_scores)) if window_scores else 0.0
        new_b = dict(b)
        new_b["mean_confusion"] = mean_conf
        new_bins.append(new_b)
    return new_bins


def compute_candidates(bins: list[dict], z_thresh: float = 1.0,
                       flag_unknown: bool = True) -> list[dict]:
    scores = [b["mean_confusion"] for b in bins]
    mu = statistics.mean(scores) if scores else 0.0
    sigma = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    candidates = []
    for b in bins:
        strat = b.get("strategy", {})
        quality = strat.get("dominant_quality", "unknown")
        is_risky = (
            strat.get("low_quality") or
            strat.get("high_pressure") or
            quality == "low" or
            (flag_unknown and quality == "unknown")
        )
        z = (b["mean_confusion"] - mu) / sigma if sigma > 0 else 0.0
        if b["mean_confusion"] > mu + z_thresh * sigma and is_risky:
            candidates.append({**b, "confusion_z": z})
    # fallback if no vision signal
    if not candidates and all(b["mean_confusion"] == 0 for b in bins):
        candidates = [
            {**b, "confusion_z": 0.0} for b in bins
            if b.get("strategy", {}).get("dominant_quality") in ("low", "unknown")
        ]
    return candidates


# ── Local Claude fusion (same logic as run_local_fusion.py) ──────────────────

ANSWER_GIVING_STRONG = [
    r"\bit(?:'s| is| will be| equals?)\b.{0,50}\b(degrees?|cm|mm|meters?|percent|%|kg|newton|joule|watt)\b",
    r"\bthe answer is\b",
    r"\bit(?:'s| is)\s+\d+[\.,]?\d*\b",
    r"\bso\s+(?:it'?s?|that'?s?|this is)\s+\d+",
    r"\bthe formula is\b",
    r"\bthe rule is\b",
    r"\balways\s+(?:equals?|is)\s+\d+",
    r"\bno matter what.{0,60}it (?:will |is )?(?:always )?\d+",
]
ANSWER_GIVING_WEAK = [
    r"\bwe(?:'re| are) going to (memorize|remember|just write down)\b",
    r"\bjust\s+(memorize|copy down)\b",
    r"\bdoesn'?t\s+(?:matter|change)[^.]{0,40}(?:always|same|fixed)",
]
MANAGEMENT_PATTERNS = [
    r"\b(sit|sitting|seat|come in|coming in|settle|quiet|silence|shh)\b",
    r"\b(equipment|projector|computer|laptop|screen|worksheet|paper|sheet)\b",
    r"\b(line up|pack up|put away|take out|open your|close your book)\b",
    r"\b(login|log in|click|open the|save|download|upload|file|folder)\b",
]
ELICITING_PATTERNS = [
    r"\?",
    r"\bwhat (do you|did you|did|does|is|are)\b",
    r"\bhow (do you|did|does|would)\b",
    r"\bwhy (do you|did|does|is|are)\b",
    r"\bcan you (explain|tell|show|describe)\b",
    r"\bwhat (do you think|would happen|notice)\b",
    r"\b(anyone|somebody|who can|who knows)\b",
    r"\btell me\b", r"\bwhat about\b",
    r"\bwhat you think\b", r"\bwhat you notice\b",
    r"\b(discover|figure out|find out|explore)\b",
    r"\byour (idea|prediction|hypothesis|guess)\b",
    r"\bdo you (think|notice|see|understand)\b",
]
GOOD_MOVES = {"Pressing for Reasoning", "Revoicing", "Keeping Everyone Together"}

def _hits(patterns, text):
    tl = text.lower()
    return sum(1 for p in patterns if re.search(p, tl))

def _first_match(patterns, text):
    tl = text.lower()
    for p in patterns:
        m = re.search(p, tl)
        if m:
            return m.group(0)[:60]
    return ""

def analyze_bin(quality, moves, excerpt):
    excerpt = (excerpt or "").strip()
    if any(m in GOOD_MOVES for m in moves):
        if _hits(ANSWER_GIVING_STRONG, excerpt) == 0:
            return False, "Teacher is using a high-quality eliciting move.", ""
    mgmt = _hits(MANAGEMENT_PATTERNS, excerpt)
    if mgmt >= 2:
        return False, "Bin contains classroom management or setup activity.", ""
    if len(excerpt) < 40:
        return False, "Excerpt too brief to confirm instructional friction.", ""
    elicit  = _hits(ELICITING_PATTERNS, excerpt)
    strong  = _hits(ANSWER_GIVING_STRONG, excerpt)
    weak    = _hits(ANSWER_GIVING_WEAK, excerpt)
    if elicit >= 2 and strong == 0 and weak == 0:
        return False, "Teacher is asking questions to elicit student thinking.", ""
    if strong >= 1 and mgmt < 2:
        rat = f"Teacher states answer/rule directly (matched: \"{_first_match(ANSWER_GIVING_STRONG, excerpt)}\")."
        alt = "Press for reasoning: ask students what they notice before revealing the answer."
        return True, rat, alt
    if weak >= 1 and elicit == 0 and mgmt < 2:
        rat = f"Teacher uses procedural directive without eliciting reasoning (matched: \"{_first_match(ANSWER_GIVING_WEAK, excerpt)}\")."
        alt = "Ask students to reason through the step before providing the answer."
        return True, rat, alt
    pressure_only = all(m in {"Other", "Pressing for Accuracy", "No Move"} for m in moves)
    if quality == "low" and pressure_only and elicit == 0:
        phits = _hits([
            r"\bjust\s+(do|get|answer|copy)\b",
            r"\bcome on\b", r"\bhurry\b", r"\bquickly\b",
            r"\bdon'?t\s+(?:worry|think)\b",
            r"\byou\s+(?:just|simply|only)\s+(?:need|have to|do)\b",
        ], excerpt)
        if phits >= 1 and len(excerpt) > 80:
            return True, "Teacher applies pressure without conceptual scaffolding.", \
                   "Slow down and ask a guiding question to help students reason through the step."
    return False, "Excerpt shows normal instructional activity without evidence of answer-giving.", ""


# ── Per-lesson runner ─────────────────────────────────────────────────────────

def run_lesson(lesson_id: str, grid_sec: float = 30.0, z_thresh: float = 1.0,
               sliding_window_sec: float = 30.0) -> None:
    work_dir   = VP_DIR / lesson_id
    frames_dir = work_dir / "frames"
    report_path = work_dir / "teacher_friction_report.json"

    if not frames_dir.exists() or not report_path.exists():
        print(f"  [{lesson_id}] missing frames or report, skipping")
        return

    report_json = json.loads(report_path.read_text())
    frames = load_frames(frames_dir)
    if not frames:
        print(f"  [{lesson_id}] no frames found")
        return

    print(f"  [{lesson_id}] DeepFace on {len(frames)} frames...", flush=True)
    from vision.confusion_timeline import build_confusion_timeline
    raw_conf = build_confusion_timeline(frames, enforce_detection=False)
    smoothed = sliding_window_average(raw_conf, sliding_window_sec)

    bins = rebuild_bins(report_json, smoothed, grid_sec)
    candidates = compute_candidates(bins, z_thresh)

    # Run local fusion
    fusion_results = []
    for c in candidates:
        strat   = c.get("strategy", {})
        quality = strat.get("dominant_quality", "unknown")
        moves   = strat.get("talk_moves", [])
        excerpt = strat.get("transcript_excerpt", "")
        friction, rationale, alt = analyze_bin(quality, moves, excerpt)
        fusion_results.append({
            "t_start": c["t_start"],
            "t_end":   c["t_end"],
            "mean_confusion": c["mean_confusion"],
            "confusion_z":    c.get("confusion_z", 0.0),
            "friction":       friction,
            "rationale":      rationale,
            "alternative_strategy": alt,
        })

    n_friction = sum(1 for r in fusion_results if r["friction"])
    print(f"  [{lesson_id}] {n_friction}/{len(candidates)} confirmed  "
          f"(bins={len(bins)}, max_conf={max((b['mean_confusion'] for b in bins), default=0):.3f})")

    # Save updated report
    updated = dict(report_json)
    updated["bins"]       = bins
    updated["candidates"] = candidates
    updated["fusion"]     = fusion_results
    report_path.write_text(json.dumps(updated, indent=2, ensure_ascii=False))

    # Save summary.json (same schema as transcript runs)
    risky_bins = [b for b in bins if
                  b.get("strategy", {}).get("dominant_quality") in ("low", "unknown") or
                  b.get("strategy", {}).get("low_quality") or
                  b.get("strategy", {}).get("high_pressure")]
    summary = {
        "lesson_id": lesson_id,
        "n_bins": len(bins),
        "n_risky_bins": len(risky_bins),
        "bins": bins,
        "risky_bins": risky_bins,
    }
    (work_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lesson", default=None)
    ap.add_argument("--all",    action="store_true")
    ap.add_argument("--force",  action="store_true", help="Re-run even if summary.json exists")
    args = ap.parse_args()

    if args.lesson:
        run_lesson(args.lesson)
        return

    if not args.all:
        ap.print_help()
        return

    lessons = sorted(d.name for d in VP_DIR.iterdir() if d.is_dir())
    for lid in lessons:
        if not args.force and (VP_DIR / lid / "summary.json").exists():
            print(f"  [skip] {lid}")
            continue
        run_lesson(lid)


if __name__ == "__main__":
    main()
