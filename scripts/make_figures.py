"""Generate PDF figures for the CVPR progress report."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import cv2
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

FIGURES_DIR = ROOT / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# Figure 1: Lead classroom frame (from AU1 video)
# ──────────────────────────────────────────────
def make_lead_frame() -> None:
    video = ROOT / "data/timss/TIMSS_AU1_Exterior_Angles_POLYGON.mp4"
    out   = FIGURES_DIR / "leadclassroom.pdf"
    cap = cv2.VideoCapture(str(video))
    # Seek to ~165 s (middle of first candidate window)
    target_sec = 165.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_sec * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("  [WARN] could not read frame for lead figure")
        return
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    ax.imshow(frame_rgb)
    ax.axis("off")
    ax.set_title(
        "TIMSS AU1 — Exterior Angles (~2:45 min)\nCandidate friction window",
        fontsize=8, pad=4,
    )
    fig.tight_layout(pad=0.3)
    fig.savefig(str(out), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ──────────────────────────────────────────────
# Figure 2: Pipeline overview diagram
# ──────────────────────────────────────────────
def make_pipeline_figure() -> None:
    out = FIGURES_DIR / "pipeline_overview.pdf"

    fig, ax = plt.subplots(figsize=(9.5, 2.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    # ---- box definitions: (x_center, y_center, width, height, label, color)
    boxes = [
        (0.75, 2.1, 1.3,  0.70, "Video +\nAudio", "#4e9af1"),
        (2.55, 2.55, 1.3, 0.60, "Frame\nExtraction", "#57b894"),
        (2.55, 1.50, 1.3, 0.60, "Whisper\nASR", "#57b894"),
        (4.30, 2.55, 1.5, 0.60, "DeepFace\nEmotion", "#f0a500"),
        (4.30, 1.50, 1.5, 0.60, "Heuristic\nStrategy Tagger", "#f0a500"),
        (6.20, 2.05, 1.5, 1.20, "30s Grid\nAlignment\n+ z-filter", "#9b59b6"),
        (8.15, 2.55, 1.5, 0.60, "Qwen2.5-VL\nFusion", "#e74c3c"),
        (8.15, 1.50, 1.5, 0.60, "Teacher\nFriction Report", "#27ae60"),
    ]

    for (cx, cy, w, h, label, color) in boxes:
        rect = mpatches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.05",
            linewidth=1.2, edgecolor="white",
            facecolor=color, alpha=0.88,
        )
        ax.add_patch(rect)
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=7.2, color="white", fontweight="bold",
                linespacing=1.4)

    # ---- arrows
    arrowprops = dict(arrowstyle="-|>", color="#333333", lw=1.1)
    arrows = [
        # video -> frame extraction
        ((0.75 + 0.65, 2.1), (2.55 - 0.65, 2.55)),
        # video -> whisper
        ((0.75 + 0.65, 2.1), (2.55 - 0.65, 1.50)),
        # frame extraction -> deepface
        ((2.55 + 0.65, 2.55), (4.30 - 0.75, 2.55)),
        # whisper -> heuristic
        ((2.55 + 0.65, 1.50), (4.30 - 0.75, 1.50)),
        # deepface -> alignment
        ((4.30 + 0.75, 2.55), (6.20 - 0.75, 2.30)),
        # heuristic -> alignment
        ((4.30 + 0.75, 1.50), (6.20 - 0.75, 1.80)),
        # alignment -> qwen
        ((6.20 + 0.75, 2.30), (8.15 - 0.75, 2.55)),
        # alignment -> report
        ((6.20 + 0.75, 1.80), (8.15 - 0.75, 1.50)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=arrowprops)

    fig.tight_layout(pad=0.1)
    fig.savefig(str(out), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ──────────────────────────────────────────────
# Figure 3: Confusion timeline (per-bin bar chart)
# ──────────────────────────────────────────────
def make_confusion_timeline() -> None:
    out  = FIGURES_DIR / "confusion_timeline_au1.pdf"
    data = json.loads((ROOT / "runs/au1_z03_v2/teacher_friction_report.json").read_text())
    bins   = data["bins"]
    t_mids = [(b["t_start"] + b["t_end"]) / 2 for b in bins]
    scores = [b["mean_confusion"] for b in bins]
    mu     = statistics.mean(scores)
    sigma  = statistics.pstdev(scores)
    thresh = mu + 0.3 * sigma

    fig, ax = plt.subplots(figsize=(6.0, 2.5))
    colors = ["#d62728" if s > thresh else "#1f77b4" for s in scores]
    ax.bar(t_mids, scores, width=27, color=colors, alpha=0.82,
           edgecolor="white", linewidth=0.4)
    ax.axhline(thresh, color="#d62728", linewidth=1.2, linestyle="--")
    ax.axhline(mu,     color="grey",    linewidth=0.8, linestyle=":")
    for b in bins:
        if b["mean_confusion"] > thresh:
            ax.axvspan(b["t_start"], b["t_end"], alpha=0.10, color="#d62728")

    blue_p  = mpatches.Patch(color="#1f77b4", alpha=0.82, label="Normal bin")
    red_p   = mpatches.Patch(color="#d62728", alpha=0.82, label="Flagged candidate")
    thr_l   = plt.Line2D([0], [0], color="#d62728", lw=1.2,
                          linestyle="--", label=f"Threshold $\\mu+0.3\\sigma={thresh:.2f}$")
    mu_l    = plt.Line2D([0], [0], color="grey", lw=0.8,
                          linestyle=":", label=f"$\\mu_C={mu:.2f}$")
    ax.legend(handles=[blue_p, red_p, thr_l, mu_l], fontsize=6.8, loc="upper left",
              ncol=2, framealpha=0.7)

    ax.set_xlabel("Time (s)", fontsize=8.5)
    ax.set_ylabel("Mean confusion score", fontsize=8.5)
    ax.set_title("Per-bin confusion timeline — AU1 Exterior Angles (first 5 min)", fontsize=8.5)
    ax.set_xlim(0, max(b["t_end"] for b in bins))
    ax.set_ylim(0, 0.85)
    ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    fig.savefig(str(out), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ──────────────────────────────────────────────
# Figure 4: Qualitative friction panels
# ──────────────────────────────────────────────
def make_qualitative_panels() -> None:
    out  = FIGURES_DIR / "qualitative_friction.pdf"
    data = json.loads((ROOT / "runs/au1_z03_v2/teacher_friction_report.json").read_text())
    fusion = data["fusion"]  # list of 3 candidates

    video = ROOT / "data/timss/TIMSS_AU1_Exterior_Angles_POLYGON.mp4"
    cap   = cv2.VideoCapture(str(video))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0

    n = min(3, len(fusion))
    fig, axes = plt.subplots(2, n, figsize=(9.5, 4.2),
                              gridspec_kw={"height_ratios": [3, 1.6]})
    if n == 1:
        axes = [[axes[0]], [axes[1]]]

    for col, entry in enumerate(fusion[:n]):
        t_mid = (entry["t_start"] + entry["t_end"]) / 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_mid * fps))
        ok, frame = cap.read()
        if ok:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            axes[0][col].imshow(frame_rgb)
        else:
            axes[0][col].set_facecolor("#eeeeee")
            axes[0][col].text(0.5, 0.5, "(frame unavailable)",
                               ha="center", va="center",
                               transform=axes[0][col].transAxes, fontsize=7)
        axes[0][col].axis("off")
        axes[0][col].set_title(
            f"Window {col+1}: {entry['t_start']:.0f}–{entry['t_end']:.0f} s",
            fontsize=8, pad=3,
        )

        # find matching bin for strategy info
        matching = next(
            (b for b in data["bins"]
             if b["t_start"] == entry["t_start"]), None
        )
        if matching:
            strat = matching["strategy"]
            excerpt = strat.get("transcript_excerpt", "")[:180].replace("\n", " ")
            quality = strat.get("dominant_quality", "?")
            confusion = matching["mean_confusion"]
            info = (
                f"Confusion: {confusion:.2f}   Quality: {quality}\n"
                f"Transcript: \"{excerpt}\"\n"
                f"Qwen verdict: {entry.get('rationale', '(pending GPU run)')}"
            )
        else:
            info = f"Rationale: {entry.get('rationale', '—')}"

        axes[1][col].axis("off")
        axes[1][col].text(
            0.03, 0.95, info,
            transform=axes[1][col].transAxes,
            fontsize=6.2, va="top", ha="left",
            wrap=True,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f9f0e8",
                      edgecolor="#ccaa88", alpha=0.9),
        )

    cap.release()
    fig.suptitle("Qualitative friction report — AU1 Exterior Angles", fontsize=9, y=1.01)
    fig.tight_layout(h_pad=0.3)
    fig.savefig(str(out), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


if __name__ == "__main__":
    print("Generating figures...")
    make_lead_frame()
    make_pipeline_figure()
    make_confusion_timeline()
    make_qualitative_panels()
    print("Done.")
