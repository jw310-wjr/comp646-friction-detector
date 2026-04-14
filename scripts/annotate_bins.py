#!/usr/bin/env python3
"""
Interactive human annotation tool for friction bin labeling.

Usage
-----
# Start or resume annotation for AU1:
python scripts/annotate_bins.py --lesson M-AU1

# After watching the video, run with --video to show timestamps
python scripts/annotate_bins.py --lesson M-AU1 --video path/to/AU1.mp4

Annotation is saved to  runs/annotations/<lesson_id>/human_labels.csv
and can be interrupted + resumed at any time (already-labeled bins are skipped).

Label guide
-----------
  1 = Friction   (risky strategy AND student confusion visible)
  0 = No friction
  s = Skip (unsure)
  q = Quit and save

After annotation, run:
  python scripts/annotate_bins.py --eval --lesson M-AU1
to compute Precision / Recall / F1 against the heuristic and ELECTRA pipelines.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RUNS_DIR      = ROOT / "runs"
TRANSCRIPT_H  = RUNS_DIR / "transcripts"
TRANSCRIPT_E  = RUNS_DIR / "transcripts_electra"
ANNOT_DIR     = RUNS_DIR / "annotations"

FIELDNAMES = ["lesson_id", "t_start", "t_end", "human_label",
              "heuristic_risky", "electra_risky", "excerpt"]


def fmt_ts(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def load_summary(lesson_id: str, run_dir: Path) -> dict | None:
    p = run_dir / lesson_id / "summary.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def annotate(lesson_id: str, video_path: str | None) -> None:
    summ_h = load_summary(lesson_id, TRANSCRIPT_H)
    summ_e = load_summary(lesson_id, TRANSCRIPT_E)

    if summ_h is None:
        sys.exit(f"Heuristic summary not found for {lesson_id}. "
                 f"Run: python scripts/run_transcript.py --lesson {lesson_id}")

    bins_h = summ_h["bins"]
    risky_h = {(b["t_start"], b["t_end"]) for b in summ_h["risky_bins"]}
    risky_e: set = set()
    if summ_e:
        risky_e = {(b["t_start"], b["t_end"]) for b in summ_e["risky_bins"]}

    out_dir = ANNOT_DIR / lesson_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "human_labels.csv"

    # Load already-labeled bins
    labeled: dict[tuple, str] = {}
    if out_csv.exists():
        with out_csv.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (float(row["t_start"]), float(row["t_end"]))
                labeled[key] = row["human_label"]

    total   = len(bins_h)
    pending = [b for b in bins_h if (b["t_start"], b["t_end"]) not in labeled]

    print(f"\n=== Human annotation: {lesson_id} ===")
    print(f"Total bins : {total}  |  Already labeled : {total - len(pending)}"
          f"  |  Remaining : {len(pending)}")
    if video_path:
        print(f"Video      : {video_path}")
    print("\nLabel: 1=Friction  0=No friction  s=Skip  q=Quit\n")

    with out_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if out_csv.stat().st_size == 0:
            writer.writeheader()

        for i, b in enumerate(pending):
            key = (b["t_start"], b["t_end"])
            ts  = f"{fmt_ts(b['t_start'])} – {fmt_ts(b['t_end'])}"
            hr  = "RISKY" if key in risky_h else "ok   "
            er  = "RISKY" if key in risky_e else "ok   "
            ex  = b.get("excerpt", "")[:120]

            print(f"[{i+1}/{len(pending)}]  {ts}   heuristic={hr}  electra={er}")
            print(f"  Excerpt: {ex}")
            if video_path:
                print(f"  -> Watch at {fmt_ts(b['t_start'])} in the video")

            while True:
                raw = input("  Label [1/0/s/q]: ").strip().lower()
                if raw in ("1", "0", "s", "q"):
                    break
                print("  Please enter 1, 0, s, or q.")

            if raw == "q":
                print("Saved progress. Quit.")
                return

            writer.writerow({
                "lesson_id":       lesson_id,
                "t_start":         b["t_start"],
                "t_end":           b["t_end"],
                "human_label":     raw,
                "heuristic_risky": 1 if key in risky_h else 0,
                "electra_risky":   1 if key in risky_e else 0,
                "excerpt":         ex,
            })
            f.flush()
            labeled[key] = raw
            print()

    print(f"\nDone! Labels saved to {out_csv}")


def evaluate(lesson_id: str) -> None:
    out_csv = ANNOT_DIR / lesson_id / "human_labels.csv"
    if not out_csv.exists():
        sys.exit(f"No annotations found. Run annotation first.")

    rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    labeled = [r for r in rows if r["human_label"] in ("0", "1")]

    if not labeled:
        print("No definitive labels (only skips). Nothing to evaluate.")
        return

    def prf(preds: list[int], golds: list[int]) -> tuple[float, float, float]:
        tp = sum(p == 1 and g == 1 for p, g in zip(preds, golds))
        fp = sum(p == 1 and g == 0 for p, g in zip(preds, golds))
        fn = sum(p == 0 and g == 1 for p, g in zip(preds, golds))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return prec, rec, f1

    gold   = [int(r["human_label"]) for r in labeled]
    h_pred = [int(r["heuristic_risky"]) for r in labeled]
    e_pred = [int(r["electra_risky"])   for r in labeled]

    hp, hr, hf = prf(h_pred, gold)
    ep, er, ef = prf(e_pred, gold)

    n_pos = sum(gold)
    print(f"\n=== Evaluation: {lesson_id} ===")
    print(f"Labeled bins : {len(labeled)}  |  Human friction=1 : {n_pos}")
    print(f"\n{'Annotator':<12} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("-" * 32)
    print(f"{'Heuristic':<12} {hp:>6.3f} {hr:>6.3f} {hf:>6.3f}")
    print(f"{'ELECTRA':<12} {ep:>6.3f} {er:>6.3f} {ef:>6.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Human annotation for friction bins")
    ap.add_argument("--lesson", required=True, help="e.g. M-AU1")
    ap.add_argument("--video",  default=None,  help="Path to .mp4 for timestamp reference")
    ap.add_argument("--eval",   action="store_true", help="Compute P/R/F1 from existing labels")
    args = ap.parse_args()

    if args.eval:
        evaluate(args.lesson)
    else:
        annotate(args.lesson, args.video)


if __name__ == "__main__":
    main()
