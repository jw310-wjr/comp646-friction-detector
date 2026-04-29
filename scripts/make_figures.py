"""Generate PDF figures for the CVPR progress report.

Figures
-------
1. pipeline_overview.pdf     — two-row pipeline diagram (Vision / Language streams)
2. risky_by_country.pdf      — % risky bins per country, math vs science
3. strategy_heatmap.pdf      — strategy-quality heatmap across all lessons
4. strategy_timeline_au1.pdf — annotated per-bin quality + talk-move timeline for AU1
5. electra_vs_heuristic.pdf  — annotator comparison by country
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
import matplotlib.ticker as ticker
import numpy as np

FIGURES_DIR             = ROOT / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_RUNS         = ROOT / "runs" / "transcripts"
TRANSCRIPT_RUNS_ELECTRA = ROOT / "runs" / "transcripts_electra"
FUSION_RUNS             = ROOT / "runs" / "claude_fusion"

COUNTRY_LABELS = {
    "AU": "Australia", "CZ": "Czech Rep.", "HK": "Hong Kong",
    "JP": "Japan",     "NL": "Netherlands","SW": "Sweden",   "US": "United States",
}
COUNTRY_ORDER = ["AU", "CZ", "HK", "JP", "NL", "SW", "US"]

QUAL_COLORS = {"high": "#27ae60", "low": "#e74c3c", "unknown": "#95a5a6"}

plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ──────────────────────────────────────────────────────────────
# Figure 1: Pipeline overview — two-row design
# ──────────────────────────────────────────────────────────────
def make_pipeline_figure() -> None:
    out = FIGURES_DIR / "pipeline_overview.pdf"

    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)
    ax.axis("off")

    # ── stream labels ──────────────────────────────────────────
    for y, label, color in [(2.95, "Vision\nStream", "#2980b9"),
                             (1.05, "Language\nStream", "#8e44ad")]:
        ax.text(0.38, y, label, ha="center", va="center", fontsize=8,
                color="white", fontweight="bold", linespacing=1.3,
                bbox=dict(boxstyle="round,pad=0.25", fc=color, ec="none"))

    # ── shared input ──────────────────────────────────────────
    _box(ax, 1.25, 2.0, 1.1, 1.6, "Video +\nAudio", "#34495e",
         sub="TIMSS\n54 lessons")

    # ── vision row ────────────────────────────────────────────
    _box(ax, 3.3,  3.0, 1.3, 0.72, "Frame\nExtraction", "#2980b9", sub="2 Hz")
    _box(ax, 5.05, 3.0, 1.5, 0.72, "DeepFace\nEmotion", "#2980b9", sub="7 classes → $c_t$  (Eq. 1)")
    _box(ax, 6.95, 3.0, 1.5, 0.72, "30-s Smoothing", "#2980b9", sub="trailing mean → $C(t)$")

    # ── language row ──────────────────────────────────────────
    _box(ax, 3.3,  1.0, 1.3, 0.72, "ASR /\nTranscript", "#8e44ad", sub="Whisper / .txt")
    _box(ax, 5.05, 1.0, 1.5, 0.72, "Talk Move\nAnnotator", "#8e44ad", sub="ELECTRA → quality")
    _box(ax, 6.95, 1.0, 1.5, 0.72, "Bin Aggregation", "#8e44ad", sub="dominant quality $q_k$")

    # ── fusion / output ───────────────────────────────────────
    _box(ax, 8.9,  2.0, 1.45, 1.6, "z-Score\nFilter\n(Eq. 2)", "#c0392b",
         sub="candidates")
    _box(ax, 10.55, 2.0, 1.0, 1.6, "Claude\nLLM\nFusion", "#16a085",
         sub="friction\nreport")

    # ── arrows ────────────────────────────────────────────────
    ap = dict(arrowstyle="-|>", color="#444", lw=1.1,
              connectionstyle="arc3,rad=0")
    # input → both streams
    _arrow(ax, 1.80, 2.55, 2.65, 3.0,  ap)
    _arrow(ax, 1.80, 1.45, 2.65, 1.0,  ap)
    # vision chain
    _arrow(ax, 3.95, 3.0,  4.30, 3.0,  ap)
    _arrow(ax, 5.80, 3.0,  6.20, 3.0,  ap)
    # language chain
    _arrow(ax, 3.95, 1.0,  4.30, 1.0,  ap)
    _arrow(ax, 5.80, 1.0,  6.20, 1.0,  ap)
    # both streams → z-filter
    _arrow(ax, 7.70, 3.0,  8.17, 2.55, ap)
    _arrow(ax, 7.70, 1.0,  8.17, 1.45, ap)
    # z-filter → Claude
    _arrow(ax, 9.62, 2.0, 10.05, 2.0,  ap)

    fig.savefig(str(out), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved {out.name}")


def _box(ax, cx, cy, w, h, label, color, sub=""):
    rect = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.06", lw=1.0,
        edgecolor="white", facecolor=color, alpha=0.92, zorder=2)
    ax.add_patch(rect)
    offset = 0.08 if sub else 0
    ax.text(cx, cy + offset, label, ha="center", va="center",
            fontsize=8.0, color="white", fontweight="bold",
            linespacing=1.35, zorder=3)
    if sub:
        ax.text(cx, cy - 0.22, sub, ha="center", va="center",
                fontsize=6.0, color="white", alpha=0.88,
                linespacing=1.2, zorder=3)


def _arrow(ax, x0, y0, x1, y1, ap):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=ap, zorder=1)


# ──────────────────────────────────────────────────────────────
# Figure 2: % risky bins per country (math vs science)
# ──────────────────────────────────────────────────────────────
def make_risky_by_country() -> None:
    agg_path = TRANSCRIPT_RUNS / "aggregate_results.json"
    if not agg_path.exists():
        print("  [SKIP] aggregate_results.json not found")
        return

    agg = json.loads(agg_path.read_text())
    out = FIGURES_DIR / "risky_by_country.pdf"

    by_country: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"math": [], "science": []})
    for row in agg:
        by_country[row["country"]][row["subject"]].append(row["pct_risky"])

    countries  = [c for c in COUNTRY_ORDER if c in by_country]
    math_means = [statistics.mean(by_country[c]["math"]) if by_country[c]["math"] else 0 for c in countries]
    sci_means  = [statistics.mean(by_country[c]["science"]) if by_country[c]["science"] else 0 for c in countries]
    math_stds  = [statistics.pstdev(by_country[c]["math"]) if len(by_country[c]["math"]) > 1 else 0 for c in countries]
    sci_stds   = [statistics.pstdev(by_country[c]["science"]) if len(by_country[c]["science"]) > 1 else 0 for c in countries]

    x     = np.arange(len(countries))
    width = 0.35
    max_val = max(max(math_means), max(sci_means)) + max(max(math_stds), max(sci_stds))

    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    bars_m = ax.bar(x - width/2, math_means, width, yerr=math_stds,
                    label="Math", color="#3498db", alpha=0.90,
                    capsize=3, error_kw={"linewidth": 0.9, "ecolor": "#2c3e50"})
    bars_s = ax.bar(x + width/2, sci_means, width, yerr=sci_stds,
                    label="Science", color="#e67e22", alpha=0.90,
                    capsize=3, error_kw={"linewidth": 0.9, "ecolor": "#2c3e50"})

    # value labels on bars
    for bar in list(bars_m) + list(bars_s):
        h = bar.get_height()
        if h > 2:
            ax.text(bar.get_x() + bar.get_width()/2, h + 1.0,
                    f"{h:.0f}%", ha="center", va="bottom", fontsize=6.5, color="#2c3e50")

    ax.set_xticks(x)
    ax.set_xticklabels([COUNTRY_LABELS.get(c, c) for c in countries], fontsize=8.5)
    ax.set_ylabel("Risky 30-s bins (%)", fontsize=9)
    ax.set_ylim(0, min(max_val * 1.25 + 5, 85))
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6, linestyle="--")
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title("Heuristic friction rate by country and subject", fontsize=9, pad=6)

    fig.tight_layout()
    fig.savefig(str(out), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ──────────────────────────────────────────────────────────────
# Figure 3: Strategy quality heatmap
# ──────────────────────────────────────────────────────────────
def make_strategy_heatmap() -> None:
    out  = FIGURES_DIR / "strategy_heatmap.pdf"
    sums = sorted(TRANSCRIPT_RUNS.glob("*/summary.json"))
    if not sums:
        print("  [SKIP] no summary.json files found")
        return

    rows_math: list[dict] = []
    rows_sci:  list[dict] = []
    for p in sums:
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

    def _draw(ax, rows, title):
        labels = [r["lesson_id"].split("-", 1)[1] for r in rows]
        data   = np.array([[r["high"], r["unknown"], r["low"]] for r in rows])
        im     = ax.imshow(data, aspect="auto", cmap="RdYlGn",
                           vmin=0, vmax=100, interpolation="nearest")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["High", "Unknown", "Low"], fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_title(title, fontsize=9, pad=4)
        for i, row in enumerate(data):
            for j, val in enumerate(row):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=5.5,
                        color="white" if val < 20 or val > 80 else "black")
        return im

    n_rows  = max(len(rows_math), len(rows_sci))
    fig_h   = max(4.0, n_rows * 0.27 + 1.2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, fig_h),
                                    gridspec_kw={"wspace": 0.45})
    im = _draw(ax1, rows_math, "Math lessons")
    _draw(ax2, rows_sci,  "Science lessons")

    cbar = fig.colorbar(im, ax=[ax1, ax2], orientation="horizontal",
                        fraction=0.028, pad=0.10, aspect=40)
    cbar.set_label("% of 30-s bins", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.suptitle("Strategy quality distribution (heuristic annotator, % per bin type)",
                 fontsize=9.5, y=1.01)
    fig.savefig(str(out), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ──────────────────────────────────────────────────────────────
# Figure 4: Strategy quality timeline — AU1 (compact, informative)
# ──────────────────────────────────────────────────────────────
def make_strategy_timeline(lesson_id: str = "M-AU1") -> None:
    src = TRANSCRIPT_RUNS / lesson_id / "summary.json"
    if not src.exists():
        print(f"  [SKIP] {lesson_id}/summary.json not found")
        return
    out = FIGURES_DIR / f"strategy_timeline_{lesson_id.replace('-','').lower()}.pdf"

    d        = json.loads(src.read_text())
    bins     = d["bins"]
    risky    = {(r["t_start"], r["t_end"]) for r in d["risky_bins"]}

    # Load Claude fusion results if available
    fusion_file = FUSION_RUNS / lesson_id / "fusion_heuristic.json"
    confirmed: set[tuple] = set()
    if fusion_file.exists():
        fdata = json.loads(fusion_file.read_text())
        confirmed = {(r["t_start"], r["t_end"])
                     for r in fdata if r.get("friction") is True}

    t_mids = [( b["t_start"] + b["t_end"]) / 2 / 60 for b in bins]  # minutes
    t_left = [b["t_start"] / 60 for b in bins]
    t_w    = [(b["t_end"] - b["t_start"]) / 60 * 0.92 for b in bins]
    colors = [QUAL_COLORS[b["dominant_quality"]] for b in bins]
    max_t  = max(b["t_end"] for b in bins) / 60

    # Abbreviate talk move labels
    def _abbrev(moves):
        abbrevs = {
            "Keeping Everyone Together": "KET", "Revoicing": "Rev",
            "Pressing for Reasoning": "PfR", "Pressing for Accuracy": "PfA",
            "No Move": "—", "Other": "Oth", "": "—",
        }
        if not moves:
            return "—"
        return "/".join(abbrevs.get(m, m[:4]) for m in moves[:2])

    labels = [_abbrev(b.get("talk_moves", [])) for b in bins]

    fig, ax = plt.subplots(figsize=(8.0, 2.4))

    ax.barh([0]*len(t_mids), t_w, left=t_left, color=colors,
            alpha=0.85, edgecolor="white", linewidth=0.4, height=0.65)

    # Heuristic risky: faint red background
    for b in bins:
        if (b["t_start"], b["t_end"]) in risky:
            ax.axvspan(b["t_start"]/60, b["t_end"]/60,
                       ymin=0.0, ymax=1.0, alpha=0.12,
                       color="#c0392b", zorder=0)

    # Claude-confirmed friction: red triangle above bar
    for b in bins:
        if (b["t_start"], b["t_end"]) in confirmed:
            cx = (b["t_start"] + b["t_end"]) / 2 / 60
            ax.plot(cx, 0.52, marker="v", color="#c0392b",
                    markersize=7, zorder=5, clip_on=False)

    # Talk-move labels inside bars
    for i, b in enumerate(bins):
        cx = t_mids[i]
        lbl = labels[i]
        ax.text(cx, 0, lbl, ha="center", va="center",
                fontsize=4.8, color="white", fontweight="bold", zorder=4)

    # Legend
    patches = [mpatches.Patch(color=v, label=k.capitalize())
               for k, v in QUAL_COLORS.items()]
    patches += [
        mpatches.Patch(color="#c0392b", alpha=0.18, label="Heuristic candidate"),
        plt.Line2D([0], [0], marker="v", color="#c0392b",
                   linestyle="none", markersize=6, label="Claude: friction"),
    ]
    ax.legend(handles=patches, fontsize=6.8, loc="upper right",
              ncol=5, framealpha=0.9, handlelength=1.2)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{int(v):d}:{int((v % 1)*60):02d}"))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.set_xlabel("Lesson time (min:s)", fontsize=8.5)
    ax.set_xlim(0, max_t)
    ax.set_yticks([])
    ax.set_ylim(-0.5, 0.8)
    n_risky = d["n_risky_bins"]
    n_conf  = len(confirmed)
    ax.set_title(
        f"Strategy quality timeline — {lesson_id}  "
        f"({n_risky}/{d['n_bins']} heuristic candidates; "
        f"{n_conf} confirmed by Claude)",
        fontsize=8.5, pad=4)

    fig.tight_layout(pad=0.4)
    fig.savefig(str(out), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ──────────────────────────────────────────────────────────────
# Figure 5: ELECTRA vs Heuristic comparison
# ──────────────────────────────────────────────────────────────
def make_electra_vs_heuristic() -> None:
    agg_heur = TRANSCRIPT_RUNS / "aggregate_results.json"
    agg_elec = TRANSCRIPT_RUNS_ELECTRA / "aggregate_results.json"
    if not agg_heur.exists() or not agg_elec.exists():
        print("  [SKIP] electra_vs_heuristic — missing aggregate_results.json")
        return

    heur_rows  = json.loads(agg_heur.read_text())
    elec_rows  = json.loads(agg_elec.read_text())
    heur_by_id = {r["lesson_id"]: r for r in heur_rows}
    elec_by_id = {r["lesson_id"]: r for r in elec_rows}
    common_ids = sorted(set(heur_by_id) & set(elec_by_id))
    if not common_ids:
        print("  [SKIP] no common lesson IDs")
        return

    by_country: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"heuristic": [], "electra": []})
    for lid in common_ids:
        c = heur_by_id[lid]["country"]
        by_country[c]["heuristic"].append(heur_by_id[lid]["pct_risky"])
        by_country[c]["electra"].append(elec_by_id[lid]["pct_risky"])

    countries   = [c for c in COUNTRY_ORDER if c in by_country]
    heur_means  = [statistics.mean(by_country[c]["heuristic"]) for c in countries]
    elec_means  = [statistics.mean(by_country[c]["electra"])   for c in countries]
    heur_stds   = [statistics.pstdev(by_country[c]["heuristic"])
                   if len(by_country[c]["heuristic"]) > 1 else 0 for c in countries]
    elec_stds   = [statistics.pstdev(by_country[c]["electra"])
                   if len(by_country[c]["electra"]) > 1 else 0 for c in countries]

    x     = np.arange(len(countries))
    width = 0.35
    max_val = max(max(heur_means), max(elec_means)) + 10

    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    bars_h = ax.bar(x - width/2, heur_means, width, yerr=heur_stds,
                    label="Keyword heuristic", color="#3498db", alpha=0.90,
                    capsize=3, error_kw={"linewidth": 0.9})
    bars_e = ax.bar(x + width/2, elec_means, width, yerr=elec_stds,
                    label="ELECTRA Talk Moves", color="#e74c3c", alpha=0.90,
                    capsize=3, error_kw={"linewidth": 0.9})

    for bar in list(bars_h) + list(bars_e):
        h = bar.get_height()
        if h > 2:
            ax.text(bar.get_x() + bar.get_width()/2, h + 1.2,
                    f"{h:.0f}%", ha="center", va="bottom",
                    fontsize=6.5, color="#2c3e50")

    ax.set_xticks(x)
    ax.set_xticklabels([COUNTRY_LABELS.get(c, c) for c in countries], fontsize=8.5)
    ax.set_ylabel("Risky 30-s bins (%)", fontsize=9)
    ax.set_ylim(0, min(max_val, 100))
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6, linestyle="--")
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title(
        f"Keyword heuristic vs.\ ELECTRA Talk Moves (Pearson $r=-0.35$, "
        f"$n={len(common_ids)}$ lessons)",
        fontsize=8.5, pad=6)

    out = FIGURES_DIR / "electra_vs_heuristic.pdf"
    fig.tight_layout()
    fig.savefig(str(out), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ──────────────────────────────────────────────────────────────
# Figure 6: Heuristic risky vs Claude-confirmed friction
# ──────────────────────────────────────────────────────────────
def make_heuristic_vs_claude() -> None:
    agg_path = TRANSCRIPT_RUNS / "aggregate_results.json"
    if not agg_path.exists():
        print("  [SKIP] heuristic_vs_claude — missing aggregate_results.json")
        return

    heur_rows = json.loads(agg_path.read_text())
    heur_by_id = {r["lesson_id"]: r for r in heur_rows}

    # Count Claude friction=True per lesson from fusion_heuristic.json
    by_country: dict[str, dict[str, list]] = defaultdict(
        lambda: {"heuristic": [], "claude": [], "precision": []})

    for lid, hr in heur_by_id.items():
        fusion_file = FUSION_RUNS / lid / "fusion_heuristic.json"
        if not fusion_file.exists():
            continue
        fdata = json.loads(fusion_file.read_text())
        n_risky = hr["n_risky_bins"]
        n_confirmed = sum(1 for r in fdata if r.get("friction") is True)
        if n_risky == 0:
            continue
        precision = 100 * n_confirmed / n_risky
        c = hr["country"]
        by_country[c]["heuristic"].append(n_risky)
        by_country[c]["claude"].append(n_confirmed)
        by_country[c]["precision"].append(precision)

    if not by_country:
        print("  [SKIP] heuristic_vs_claude — no fusion results found")
        return

    countries     = [c for c in COUNTRY_ORDER if c in by_country]
    prec_means    = [statistics.mean(by_country[c]["precision"]) for c in countries]
    prec_stds     = [statistics.pstdev(by_country[c]["precision"])
                     if len(by_country[c]["precision"]) > 1 else 0 for c in countries]
    heur_totals   = [sum(by_country[c]["heuristic"]) for c in countries]
    claude_totals = [sum(by_country[c]["claude"])    for c in countries]

    x = np.arange(len(countries))
    width = 0.38
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.2),
                                    gridspec_kw={"wspace": 0.38})

    # Left: precision bar per country
    bars = ax1.bar(x, prec_means, width * 1.6, yerr=prec_stds,
                   color="#8e44ad", alpha=0.85, capsize=3,
                   error_kw={"linewidth": 0.9})
    for bar in bars:
        h = bar.get_height()
        if h > 2:
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
                     f"{h:.0f}%", ha="center", va="bottom", fontsize=6.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([COUNTRY_LABELS.get(c, c) for c in countries], fontsize=8)
    ax1.set_ylabel("Claude precision (%)\n(confirmed / heuristic risky)", fontsize=8)
    ax1.set_ylim(0, 80)
    ax1.grid(axis="y", alpha=0.25, linewidth=0.6, linestyle="--")
    ax1.set_title("Heuristic→Claude confirmation rate by country", fontsize=8.5, pad=4)

    # Right: scatter heuristic_risky vs claude_confirmed per lesson
    all_h, all_c, all_col = [], [], []
    cmap_countries = plt.colormaps["tab10"]
    country_idx = {c: i for i, c in enumerate(countries)}
    for lid, hr in heur_by_id.items():
        fusion_file = FUSION_RUNS / lid / "fusion_heuristic.json"
        if not fusion_file.exists():
            continue
        fdata = json.loads(fusion_file.read_text())
        n_risky = hr["n_risky_bins"]
        n_confirmed = sum(1 for r in fdata if r.get("friction") is True)
        if n_risky == 0:
            continue
        c = hr["country"]
        all_h.append(n_risky)
        all_c.append(n_confirmed)
        all_col.append(cmap_countries(country_idx.get(c, 0)))

    ax2.scatter(all_h, all_c, c=all_col, s=28, alpha=0.85, edgecolors="white", linewidths=0.4)
    if all_h:
        mn, mx = min(all_h), max(all_h)
        ax2.plot([mn, mx], [mn, mx], "--", color="#aaa", linewidth=0.8, label="precision=100%")
    ax2.set_xlabel("Heuristic risky bins", fontsize=8.5)
    ax2.set_ylabel("Claude friction=True", fontsize=8.5)
    ax2.set_title("Per-lesson heuristic vs Claude (each dot = 1 lesson)", fontsize=8.5, pad=4)
    legend_patches = [mpatches.Patch(color=cmap_countries(country_idx[c]),
                                      label=COUNTRY_LABELS.get(c, c))
                      for c in countries]
    ax2.legend(handles=legend_patches, fontsize=6.5, framealpha=0.9, ncol=2)
    ax2.grid(alpha=0.2, linewidth=0.5)

    out = FIGURES_DIR / "heuristic_vs_claude.pdf"
    fig.savefig(str(out), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved {out.name}")


if __name__ == "__main__":
    print("Generating figures...")
    make_pipeline_figure()
    make_risky_by_country()
    make_strategy_heatmap()
    make_strategy_timeline("M-AU1")
    make_electra_vs_heuristic()
    make_heuristic_vs_claude()
    print(f"\nAll figures → {FIGURES_DIR}")
