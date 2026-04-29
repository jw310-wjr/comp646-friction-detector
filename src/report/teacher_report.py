"""Post-session Teacher Friction Report (proposal §2 output)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from schemas import TeacherFrictionReport


def save_report(report: TeacherFrictionReport, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    (out_dir / "teacher_friction_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (out_dir / "teacher_friction_report.txt").write_text(
        render_text_summary(report),
        encoding="utf-8",
    )


def render_text_summary(r: TeacherFrictionReport) -> str:
    lines = [
        "Teacher Friction Report",
        "=======================",
        f"Video: {r.video_path}",
        f"Work directory: {r.work_dir}",
        "",
        f"Alignment bins (30 s grid): {len(r.bins)}",
        f"Heuristic candidates: {len(r.candidates)}",
        f"Multimodal fusion entries: {len(r.fusion)}",
        "",
    ]
    for i, f in enumerate(r.fusion, start=1):
        lines.append(f"--- Event {i} [{f.t_start:.1f}s – {f.t_end:.1f}s] ---")
        lines.append(f"Friction: {f.friction}")
        lines.append(f"Rationale: {f.rationale}")
        lines.append(f"Suggested move: {f.alternative_strategy}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"
