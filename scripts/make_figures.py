"""Generate PDF figures for the CVPR progress report.

Figures produced
----------------
1. pipeline_overview.pdf     — pipeline block diagram (no external data)
2. risky_by_country.pdf      — % risky bins per country, math vs science
3. strategy_heatmap.pdf      — strategy-quality distribution across all lessons
4. strategy_timeline_au1.pdf — per-bin quality timeline for AU1 (example lesson)

Figures 1–4 require only the transcript-analysis results in runs/transcripts/.
The original lead-frame and qualitative-panels figures (which need .mp4 video)
are preserved as make_lead_frame() / make_qualitative_panels() but are skipped
when video files are absent.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

FIGURES_DIR = ROOT / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_RUNS         = ROOT / "runs" / "transcripts"
TRANSCRIPT_RUNS_ELECTRA = ROOT / "runs" / "transcripts_electra"

COUNTRY_LABELS = {
    "AU": "Australia",
    "CZ": "Czech Rep.",
    "HK": "Hong Kong",
    "JP": "Japan",
    "NL": "Netherlands",
    "SW": "Sweden",
    "US": "United States",
}
COUNTRY_ORDER = ["AU", "CZ", "HK", "JP", "NL", "SW", "US"]

QUAL_COLORS = {
    "high":    "#2ecc71",
    "low":     "#e74c3c",
    "unknown": "#95a5a6",
}


# ──────────────────────────────────────────────
# Figure 1: Pipeline overview diagram
# ──────────────────────────────────────────────
def make_pipeline_figure() -> None:
    out = FIGURES_DIR / "pipeline_overview.pdf"

    fig, ax = plt.subplots(figsize=(9.5, 2.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    boxes = [
        (0.75, 2.1,  1.3, 0.70, "Video +\nAudio",              "#4e9af1"),
        (2.55, 2.55, 1.3, 0.60, "Frame\nExtraction",            "#57b894"),
        (2.55, 1.50, 1.3, 0.60, "Whisper\nASR",                 "#57b894"),
        (4.30, 2.55, 1.5, 0.60, "DeepFace\nEmotion",            "#f0a500"),
        (4.30, 1.50, 1.5, 0.60, "Heuristic\nStrategy Tagger",   "#f0a500"),
        (6.20, 2.05, 1.5, 1.20, "30s Grid\nAlignment\n+ z-filter", "#9b59b6"),
        (8.15, 2.55, 1.5, 0.60, "Claude LLM\nFusion",           "#e74c3c"),
        (8.15, 1.50, 1.5, 0.60, "Teacher\nFriction Report",     "#27ae60"),
    ]

    for cx, cy, w, h, label, color in boxes:
        rect = mpatches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.05",
            linewidth=1.2, edgecolor="white",
            facecolor=color, alpha=0.88,
        )
        ax.add_patch(rect)
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=7.2, color="white", fontweight="bold", linespacing=1.4)

    arrowprops = dict(arrowstyle="-|>", color="#333333", lw=1.1)
    arrows = [
        ((0.75 + 0.65, 2.1),  (2.55 - 0.65, 2.55)),
        ((0.75 + 0.65, 2.1),  (2.55 - 0.65, 1.50)),
        ((2.55 + 0.65, 2.55), (4.30 - 0.75, 2.55)),
        ((2.55 + 0.65, 1.50), (4.30 - 0.75, 1.50)),
        ((4.30 + 0.75, 2.55), (6.20 - 0.75, 2.30)),
        ((4.30 + 0.75, 1.50), (6.20 - 0.75, 1.80)),
        ((6.20 + 0.75, 2.30), (8.15 - 0.75, 2.55)),
        ((6.20 + 0.75, 1.80), (8.15 - 0.75, 1.50)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=arrowprops)

    fig.tight_layout(pad=0.1)
    fig.savefig(str(out), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ──────────────────────────────────────────────
# Figure 2: % risky bins per country (math vs science)
# ──────────────────────────────────────────────
def make_risky_by_country() -> None:
    agg_path = TRANSCRIPT_RUNS / "aggregate_results.json"
    if not agg_path.exists():
        print("  [SKIP] aggregate_results.json not found — run run_transcript.py first")
        return

    agg = json.loads(agg_path.read_text())
    out = FIGURES_DIR / "risky_by_country.pdf"

    by_country: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"math": [], "science": []}
    )
    for row in agg:
        by_country[row["country"]][row["subject"]].append(row["pct_risky"])

    countries = [c for c in COUNTRY_ORDER if c in by_country]
    math_means = [
        statistics.mean(by_country[c]["math"]) if by_country[c]["math"] else 0
        for c in countries
    ]
    sci_means = [
        statistics.mean(by_country[c]["science"]) if by_country[c]["science"] else 0
        for c in countries
    ]
    math_stds = [
        statistics.pstdev(by_country[c]["math"]) if len(by_country[c]["math"]) > 1 else 0
        for c in countries
    ]
    sci_stds = [
        statistics.pstdev(by_country[c]["science"]) if len(by_country[c]["science"]) > 1 else 0
        for c in countries
    ]

    x = np.arange(len(countries))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.bar(x - width / 2, math_means, width, yerr=math_stds,
           label="Math", color="#4e9af1", alpha=0.88,
           capsize=5, error_kw={"linewidth": 1.4, "ecolor": "#222222"})
    ax.bar(x + width / 2, sci_means, width, yerr=sci_stds,
           label="Science", color="#f0a500", alpha=0.88,
           capsize=5, error_kw={"linewidth": 1.4, "ecolor": "#222222"})

    ax.set_xticks(x)
    ax.set_xticklabels([COUNTRY_LABELS.get(c, c) for c in countries], fontsize=8.5)
    ax.set_ylabel("% risky 30-s bins (heuristic)", fontsize=9)
    ax.set_title(
        "Heuristic friction rate by country and subject\n"
        "(low-quality or high-pressure strategy; unknown strategy flagged)",
        fontsize=9,
    )
    ax.legend(fontsize=8.5)
    ax.set_ylim(0, 80)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    fig.savefig(str(out), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ──────────────────────────────────────────────
# Figure 3: Strategy quality heatmap (lesson × quality breakdown)
# ──────────────────────────────────────────────
def make_strategy_heatmap() -> None:
    out = FIGURES_DIR / "strategy_heatmap.pdf"
    summaries = sorted(TRANSCRIPT_RUNS.glob("*/summary.json"))
    if not summaries:
        print("  [SKIP] no summary.json files found")
        return

    rows_math: list[dict] = []
    rows_sci:  list[dict] = []
    for p in summaries:
        d = json.loads(p.read_text())
        bins = d["bins"]
        if not bins:
            continue
        total = len(bins)
        counts = {"high": 0, "low": 0, "unknown": 0}
        for b in bins:
            counts[b["dominant_quality"]] = counts.get(b["dominant_quality"], 0) + 1
        entry = {
            "lesson_id": d["lesson_id"],
            "high":    100 * counts["high"]    / total,
            "low":     100 * counts["low"]     / total,
            "unknown": 100 * counts["unknown"] / total,
        }
        (rows_math if d["lesson_id"].startswith("M-") else rows_sci).append(entry)

    def _draw(ax: plt.Axes, rows: list[dict], title: str):
        labels = [r["lesson_id"].split("-", 1)[1] for r in rows]
        data = np.array([[r["high"], r["unknown"], r["low"]] for r in rows])
        im = ax.imshow(data, aspect="auto", cmap="RdYlGn",
                       vmin=0, vmax=100, interpolation="nearest")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["High quality", "Unknown", "Low quality"], fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_title(title, fontsize=9, pad=4)
        for i, row in enumerate(data):
            for j, val in enumerate(row):
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=5.5,
                        color="black" if 20 < val < 80 else "white")
        return im

    n_rows = max(len(rows_math), len(rows_sci))
    fig_h = max(4.5, n_rows * 0.28 + 1.5)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, fig_h), gridspec_kw={"wspace": 0.5}
    )
    im = _draw(ax1, rows_math, "Math lessons")
    _draw(ax2, rows_sci, "Science lessons")

    cbar = fig.colorbar(im, ax=[ax1, ax2], orientation="horizontal",
                        fraction=0.03, pad=0.10, aspect=40)
    cbar.set_label("% of 30-s bins", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        "Strategy quality distribution by lesson (heuristic annotator)",
        fontsize=10, y=1.01,
    )
    fig.savefig(str(out), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ──────────────────────────────────────────────
# Figure 4: Per-bin strategy quality timeline (AU1 example)
# ──────────────────────────────────────────────
def make_strategy_timeline(lesson_id: str = "M-AU1") -> None:
    src = TRANSCRIPT_RUNS / lesson_id / "summary.json"
    if not src.exists():
        print(f"  [SKIP] {lesson_id}/summary.json not found")
        return
    out = FIGURES_DIR / f"strategy_timeline_{lesson_id.replace('-', '').lower()}.pdf"

    d = json.loads(src.read_text())
    bins = d["bins"]
    risky_set = {(r["t_start"], r["t_end"]) for r in d["risky_bins"]}

    t_mids  = [(b["t_start"] + b["t_end"]) / 2 for b in bins]
    colors  = [QUAL_COLORS[b["dominant_quality"]] for b in bins]
    bar_w   = (bins[0]["t_end"] - bins[0]["t_start"]) * 0.92 if bins else 28.0

    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.bar(t_mids, [1] * len(t_mids), width=bar_w,
           color=colors, alpha=0.85, edgecolor="white", linewidth=0.3)

    for b in bins:
        if (b["t_start"], b["t_end"]) in risky_set:
            ax.axvspan(b["t_start"], b["t_end"], ymin=0, ymax=1,
                       alpha=0.18, color="#d62728", zorder=0)

    patches = [
        mpatches.Patch(color=v, label=k.capitalize())
        for k, v in QUAL_COLORS.items()
    ]
    patches.append(mpatches.Patch(color="#d62728", alpha=0.25, label="Risky (shaded)"))
    ax.legend(handles=patches, fontsize=7.5, loc="upper right", ncol=4)

    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_yticks([])
    ax.set_xlim(0, max(b["t_end"] for b in bins))
    ax.set_title(
        f"Strategy quality timeline — {lesson_id} "
        f"({d['n_risky_bins']}/{d['n_bins']} bins flagged as risky)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(str(out), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ──────────────────────────────────────────────
# (Optional) Video-dependent figures — skipped if no .mp4
# ──────────────────────────────────────────────
def make_lead_frame() -> None:
    try:
        import cv2
    except ImportError:
        print("  [SKIP] lead frame — opencv not installed")
        return
    video = ROOT / "data/timss/TIMSS_AU1_Exterior_Angles_POLYGON.mp4"
    if not video.exists():
        print(f"  [SKIP] lead frame — video not found: {video.name}")
        return
    out = FIGURES_DIR / "leadclassroom.pdf"
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(165.0 * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("  [WARN] could not read lead frame")
        return
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    ax.imshow(frame_rgb)
    ax.axis("off")
    ax.set_title("TIMSS AU1 — Exterior Angles (~2:45 min)\nCandidate friction window",
                 fontsize=8, pad=4)
    fig.tight_layout(pad=0.3)
    fig.savefig(str(out), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


def make_qualitative_panels() -> None:
    """Figure 2: three AU1 bins (confirmed / false-positive / disagreement).

    Pure-matplotlib (no tabs, no Keynote) so every space is an ordinary
    space and the PDF renders cleanly in CVPR double-column width.
    """
    out = FIGURES_DIR / "qualitative_examples.pdf"

    panels = [
        {
            "title":     "Confirmed Friction",
            "title_bg":  "#c0392b",
            "timespan":  "5:00 - 5:30",
            "excerpt":   ("[T] No matter what the size of\n"
                          "the angles are, if you have a\n"
                          "five pointed star inscribed in\n"
                          "a circle it will be 180 degrees."),
            "labels": [
                ("Heuristic:",    "RISKY (low quality)",       "#c0392b"),
                ("ELECTRA:",      "RISKY (No Move, low)",      "#c0392b"),
                ("Human label:",  "FRICTION",                  "#c0392b"),
            ],
            "llm":       ("friction = True\n"
                          "\"Teacher states the answer\n"
                          "students should discover.\"\n"
                          "Alt: ask students to share\n"
                          "their measured result first."),
            "llm_bg":    "#fdecea",
        },
        {
            "title":     "Heuristic False Positive",
            "title_bg":  "#b7791f",
            "timespan":  "21:30 - 22:00",
            "excerpt":   ("[T] Just make sure ... just make\n"
                          "sure they join. Get rid of that\n"
                          "one. Use the pointer tool, just\n"
                          "to make the shape."),
            "labels": [
                ("Heuristic:",    "RISKY (high pressure)",     "#c0392b"),
                ("ELECTRA:",      "RISKY (Pressing for Accuracy)", "#c0392b"),
                ("Human label:",  "Not friction",              "#2e7d32"),
            ],
            "llm":       ("friction = False\n"
                          "\"Pressure refers to software\n"
                          "construction steps, not\n"
                          "mathematical reasoning.\""),
            "llm_bg":    "#fff7e0",
        },
        {
            "title":     "Annotator Disagreement",
            "title_bg":  "#1f618d",
            "timespan":  "8:00 - 8:30",
            "excerpt":   ("[T] Or they could be words that\n"
                          "maybe you do know, but you think\n"
                          "that is a pretty important word.\n"
                          "Just do this with a brief discussion."),
            "labels": [
                ("Heuristic:",    "RISKY (unknown, no keyword)", "#c0392b"),
                ("ELECTRA:",      "HIGH (Pressing for Reasoning)", "#2e7d32"),
                ("Human label:",  "Not friction",              "#2e7d32"),
            ],
            "llm":       ("friction = False\n"
                          "\"Teacher prompts collaborative\n"
                          "vocabulary discussion, a\n"
                          "high-quality instructional move.\""),
            "llm_bg":    "#e8f1fa",
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.4))
    for ax, panel in zip(axes, panels):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Header strip
        header = mpatches.FancyBboxPatch(
            (0.02, 0.90), 0.96, 0.08,
            boxstyle="round,pad=0.01",
            linewidth=0, facecolor=panel["title_bg"], alpha=0.92,
        )
        ax.add_patch(header)
        ax.text(0.5, 0.94, panel["title"], ha="center", va="center",
                fontsize=11.5, color="white", fontweight="bold")
        ax.text(0.5, 0.865, panel["timespan"], ha="center", va="top",
                fontsize=10, color="#333333")

        # Transcript box
        ax.text(0.04, 0.82, "Transcript excerpt:", fontsize=9.5,
                fontweight="bold", color="#222222")
        ax.text(0.04, 0.78, panel["excerpt"], fontsize=9,
                family="monospace", color="#222222",
                va="top", linespacing=1.35)

        # Label rows
        y = 0.44
        for name, value, color in panel["labels"]:
            ax.text(0.04, y, name, fontsize=9.5, fontweight="bold",
                    color="#222222", va="center")
            ax.text(0.36, y, value, fontsize=9.5, color=color,
                    fontweight="bold", va="center")
            y -= 0.055

        # LLM fusion box
        llm_box = mpatches.FancyBboxPatch(
            (0.03, 0.02), 0.94, 0.22,
            boxstyle="round,pad=0.012",
            linewidth=0.8, edgecolor="#999999",
            facecolor=panel["llm_bg"], alpha=0.95,
        )
        ax.add_patch(llm_box)
        ax.text(0.06, 0.205, "LLM fusion:", fontsize=9.5,
                fontweight="bold", color="#222222", va="top")
        ax.text(0.06, 0.175, panel["llm"], fontsize=8.5,
                color="#222222", va="top", linespacing=1.35,
                family="monospace")

    fig.tight_layout(pad=0.8)
    fig.savefig(str(out), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ──────────────────────────────────────────────
# Figure 5: ELECTRA vs Heuristic annotator comparison
# ──────────────────────────────────────────────
def make_electra_vs_heuristic() -> None:
    """Side-by-side % risky bins per country: heuristic (flag_unknown) vs ELECTRA."""
    agg_heur = TRANSCRIPT_RUNS / "aggregate_results.json"
    agg_elec = TRANSCRIPT_RUNS_ELECTRA / "aggregate_results.json"
    if not agg_heur.exists():
        print("  [SKIP] electra_vs_heuristic — heuristic aggregate_results.json not found")
        return
    if not agg_elec.exists():
        print("  [SKIP] electra_vs_heuristic — ELECTRA aggregate_results.json not found "
              "(run: python scripts/run_transcript.py --out-dir runs/transcripts_electra)")
        return

    heur_rows = json.loads(agg_heur.read_text())
    elec_rows = json.loads(agg_elec.read_text())

    # Index by lesson_id
    heur_by_id = {r["lesson_id"]: r for r in heur_rows}
    elec_by_id = {r["lesson_id"]: r for r in elec_rows}
    common_ids = sorted(set(heur_by_id) & set(elec_by_id))
    if not common_ids:
        print("  [SKIP] electra_vs_heuristic — no common lesson IDs between runs")
        return

    # Aggregate per country
    by_country: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"heuristic": [], "electra": []}
    )
    for lid in common_ids:
        country = heur_by_id[lid]["country"]
        by_country[country]["heuristic"].append(heur_by_id[lid]["pct_risky"])
        by_country[country]["electra"].append(elec_by_id[lid]["pct_risky"])

    countries = [c for c in COUNTRY_ORDER if c in by_country]
    heur_means = [statistics.mean(by_country[c]["heuristic"]) for c in countries]
    elec_means = [statistics.mean(by_country[c]["electra"])   for c in countries]
    heur_stds  = [
        statistics.pstdev(by_country[c]["heuristic"]) if len(by_country[c]["heuristic"]) > 1 else 0
        for c in countries
    ]
    elec_stds  = [
        statistics.pstdev(by_country[c]["electra"]) if len(by_country[c]["electra"]) > 1 else 0
        for c in countries
    ]

    x = np.arange(len(countries))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x - width / 2, heur_means, width, yerr=heur_stds,
           label="Heuristic (flag_unknown=True)", color="#4e9af1", alpha=0.88,
           capsize=5, error_kw={"linewidth": 1.4, "ecolor": "#222222"})
    ax.bar(x + width / 2, elec_means, width, yerr=elec_stds,
           label="ELECTRA Talk Moves (YaHi)", color="#e74c3c", alpha=0.88,
           capsize=5, error_kw={"linewidth": 1.4, "ecolor": "#222222"})

    ax.set_xticks(x)
    ax.set_xticklabels([COUNTRY_LABELS.get(c, c) for c in countries], fontsize=8.5)
    ax.set_ylabel("% risky 30-s bins", fontsize=9)
    ax.set_title(
        "Heuristic vs ELECTRA Talk Moves: friction rate by country\n"
        f"(ELECTRA: bins with all 'No Move' or low-quality strategy; n={len(common_ids)} lessons)",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    out = FIGURES_DIR / "electra_vs_heuristic.pdf"
    fig.savefig(str(out), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


if __name__ == "__main__":
    print("Generating figures...")
    make_pipeline_figure()
    make_risky_by_country()
    make_strategy_heatmap()
    make_strategy_timeline("M-AU1")
    make_electra_vs_heuristic()
    make_lead_frame()
    make_qualitative_panels()
    print(f"\nAll figures saved to: {FIGURES_DIR}")
