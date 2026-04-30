#!/usr/bin/env python3
"""
Local Claude-equivalent friction fusion — no external API calls.

Applies the same two-stage logic as ClaudeFrictionFusion but runs entirely
locally, using the judgment rules derived from the AU1 human-annotated ground
truth and the methodology in the progress report.

Genuine pedagogical friction requires ALL of:
  (a) Teacher directly states an answer/rule/result students should discover, OR
      applies pressure without conceptual support
  (b) The excerpt is instructional content (not management / transitions / setup)
  (c) No evidence the teacher is eliciting rather than telling

Usage:
    python scripts/run_local_fusion.py [--lesson M-AU2] [--all]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_HEURISTIC = ROOT / "runs" / "transcripts"
RUNS_FUSION    = ROOT / "runs" / "claude_fusion"


# ── Keyword banks ────────────────────────────────────────────────────────────

# STRONG: teacher unambiguously states an answer/rule students should discover
ANSWER_GIVING_STRONG = [
    r"\bit(?:'s| is| will be| equals?)\b.{0,50}\b(degrees?|cm|mm|meters?|percent|%|kg|newton|joule|watt)\b",
    r"\bthe answer is\b",
    r"\bit(?:'s| is)\s+\d+[\.,]?\d*\b",          # "it's 75", "it is 180"
    r"\bso\s+(?:it'?s?|that'?s?|this is)\s+\d+",  # "so it's 750"
    r"\bthe formula is\b",
    r"\bthe rule is\b",
    r"\balways\s+(?:equals?|is)\s+\d+",
    r"\bno matter what.{0,60}it (?:will |is )?(?:always )?\d+",
]

# WEAK: procedural directives (only flag if no eliciting signals present)
ANSWER_GIVING_WEAK = [
    r"\bwe(?:'re| are) going to (memorize|remember|just write down)\b",
    r"\bjust\s+(memorize|copy down)\b",
    r"\bdoesn'?t\s+(?:matter|change)[^.]{0,40}(?:always|same|fixed)",
]

ANSWER_GIVING_PATTERNS = ANSWER_GIVING_STRONG + ANSWER_GIVING_WEAK

# Signals of classroom MANAGEMENT (off-task affect, not instructional quality)
MANAGEMENT_PATTERNS = [
    r"\b(sit|sitting|seat|seats?|come in|coming in|settle|quiet|silence|shh)\b",
    r"\b(equipment|projector|computer|laptop|screen|worksheet|handout|paper|sheet)\b",
    r"\b(line up|pack up|put away|take out|open your|close your book)\b",
    r"\b(lunch|break|bell|dismiss|leaving|corridor|hallway)\b",
    r"\b(late|arrive|enter|absent)\b",
    r"\b(login|log in|click|open the|save|download|upload|file|folder)\b",   # software
]

# Signals the teacher is ELICITING / engaging (good pedagogy — NOT friction)
ELICITING_PATTERNS = [
    r"\?",                                          # question marks
    r"\bwhat (do you|did you|did|does|is|are)\b",
    r"\bhow (do you|did|does|would)\b",
    r"\bwhy (do you|did|does|is|are)\b",
    r"\bcan you (explain|tell|show|describe)\b",
    r"\bwhat (do you think|would happen|notice)\b",
    r"\b(anyone|somebody|who can|who knows)\b",
    r"\btell me\b",
    r"\bwhat about\b",
    r"\bwhat you think\b",                          # "write down what you think"
    r"\bwhat you notice\b",
    r"\b(discover|figure out|find out|explore)\b",  # guided discovery language
    r"\byour (idea|prediction|hypothesis|guess)\b",
    r"\bdo you (think|notice|see|understand)\b",
]

# Pressing for Reasoning → high-quality move (never friction on its own)
GOOD_MOVES = {"Pressing for Reasoning", "Revoicing", "Keeping Everyone Together"}


# ── Scoring functions ─────────────────────────────────────────────────────────

def _matches(patterns: list[str], text: str) -> int:
    text_lower = text.lower()
    return sum(1 for p in patterns if re.search(p, text_lower))


def analyze_bin(
    t_start: float,
    t_end: float,
    quality: str,
    moves: list[str],
    excerpt: str,
) -> dict:
    """Return fusion result dict (same schema as ClaudeFrictionFusion)."""

    excerpt = (excerpt or "").strip()

    # ── fast-reject rules ─────────────────────────────────────────────────────

    # 1. Has any good pedagogical move → very unlikely friction
    if any(m in GOOD_MOVES for m in moves):
        if _matches(ANSWER_GIVING_PATTERNS, excerpt) == 0:
            return _result(False, "Teacher is using a high-quality eliciting move.", "")

    # 2. Mostly management / off-task content
    mgmt_hits = _matches(MANAGEMENT_PATTERNS, excerpt)
    if mgmt_hits >= 2:
        return _result(False, "Bin contains classroom management or setup activity.", "")

    # 3. Short or empty excerpt — not enough evidence
    teacher_words = [l for l in excerpt.split("[T]") if l.strip()]
    if len(excerpt) < 40 or len(teacher_words) == 0:
        return _result(False, "Excerpt too brief to confirm instructional friction.", "")

    # 4. High eliciting signal
    elicit_hits = _matches(ELICITING_PATTERNS, excerpt)
    strong_hits  = _matches(ANSWER_GIVING_STRONG, excerpt)
    weak_hits    = _matches(ANSWER_GIVING_WEAK,   excerpt)

    if elicit_hits >= 2 and strong_hits == 0 and weak_hits == 0:
        return _result(False, "Teacher is asking questions to elicit student thinking.", "")

    # ── positive friction indicators ──────────────────────────────────────────

    # 5a. Strong answer-giving pattern (overrides eliciting if not just managing)
    if strong_hits >= 1 and mgmt_hits < 2:
        rationale = (
            "Teacher directly states an answer or rule students should derive "
            f"(matched: \"{_first_match(ANSWER_GIVING_STRONG, excerpt)}\")."
        )
        alt = (
            "Instead of stating the result, press for reasoning: ask students "
            "what they notice or what they predict before revealing the answer."
        )
        return _result(True, rationale, alt)

    # 5b. Weak answer-giving pattern only fires when no eliciting present
    if weak_hits >= 1 and elicit_hits == 0 and mgmt_hits < 2:
        rationale = (
            "Teacher uses procedural directive without eliciting student reasoning "
            f"(matched: \"{_first_match(ANSWER_GIVING_WEAK, excerpt)}\")."
        )
        alt = "Ask students to reason through the step before providing the answer."
        return _result(True, rationale, alt)

    # 6. Low quality, no eliciting, all "Other" or "Pressing for Accuracy" moves
    pressure_only = all(m in {"Other", "Pressing for Accuracy", "No Move"} for m in moves)
    if quality == "low" and pressure_only and elicit_hits == 0:
        # Check for rushed / pressure language without conceptual support
        pressure_hits = sum(1 for p in [
            r"\bjust\s+(do|get|write|answer|copy)\b",
            r"\bcome on\b",
            r"\bhurry\b",
            r"\bquickly\b",
            r"\bdon'?t\s+(?:worry|think)\b",
            r"\byou\s+(?:just|simply|only)\s+(?:need|have to|do)\b",
        ] if re.search(p, excerpt.lower()))
        if pressure_hits >= 1 and len(excerpt) > 80:
            rationale = "Teacher applies time or performance pressure without conceptual scaffolding."
            alt = "Slow down and ask a guiding question to help students reason through the step."
            return _result(True, rationale, alt)

    # ── default: not friction ─────────────────────────────────────────────────
    if quality == "unknown":
        return _result(False,
            "Strategy labeled unknown; excerpt shows normal instructional activity "
            "without evidence of answer-giving or counterproductive pressure.", "")
    else:
        return _result(False,
            "Low-quality strategy label, but transcript shows teacher engaging students "
            "without directly revealing answers or applying counterproductive pressure.", "")


def _result(friction: bool, rationale: str, alt: str) -> dict:
    return {"friction": friction, "rationale": rationale, "alternative_strategy": alt}


def _first_match(patterns: list[str], text: str) -> str:
    text_lower = text.lower()
    for p in patterns:
        m = re.search(p, text_lower)
        if m:
            return m.group(0)[:60]
    return ""


# ── Per-lesson runner ─────────────────────────────────────────────────────────

def run_lesson(lesson_id: str, verbose: bool = True) -> dict:
    summary_path = RUNS_HEURISTIC / lesson_id / "summary.json"
    if not summary_path.exists():
        print(f"  ERROR: {summary_path} not found")
        return {}

    out_dir = RUNS_FUSION / lesson_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fusion_heuristic.json"

    data = json.loads(summary_path.read_text())
    risky_bins = data.get("risky_bins", [])

    results = []
    for b in risky_bins:
        res = analyze_bin(
            t_start=b["t_start"],
            t_end=b["t_end"],
            quality=b["dominant_quality"],
            moves=b.get("talk_moves", []),
            excerpt=b.get("excerpt", ""),
        )
        results.append({
            "t_start": b["t_start"],
            "t_end":   b["t_end"],
            **res,
        })

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    n_friction = sum(1 for r in results if r["friction"])
    if verbose:
        print(f"  {lesson_id}: {n_friction}/{len(results)} confirmed friction")
    return {"lesson_id": lesson_id, "n_candidates": len(results), "n_confirmed": n_friction}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lesson", default=None, help="Single lesson ID, e.g. M-AU2")
    ap.add_argument("--all", action="store_true", help="Run on all lessons (skip already done)")
    ap.add_argument("--force", action="store_true", help="Re-run even if output exists")
    args = ap.parse_args()

    if args.lesson:
        run_lesson(args.lesson)
        return

    if not args.all:
        ap.print_help()
        return

    lessons = sorted(
        p.parent.name
        for p in RUNS_HEURISTIC.glob("*/summary.json")
    )

    stats = []
    for lid in lessons:
        out = RUNS_FUSION / lid / "fusion_heuristic.json"
        if out.exists() and not args.force:
            print(f"  [skip] {lid}")
            continue
        stat = run_lesson(lid)
        if stat:
            stats.append(stat)

    if stats:
        total_conf  = sum(s["n_confirmed"]  for s in stats)
        total_cand  = sum(s["n_candidates"] for s in stats)
        print(f"\nDone: {len(stats)} lessons | {total_conf}/{total_cand} confirmed friction")


if __name__ == "__main__":
    main()
