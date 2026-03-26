from __future__ import annotations

from schemas import ConfusionPoint


def sliding_window_average(
    points: list[ConfusionPoint],
    window_sec: float,
) -> list[ConfusionPoint]:
    """Trailing mean: at each time t, average scores for points with t_j in (t - window, t]."""
    if not points:
        return []
    times_scores = sorted((p.t_sec, p.score) for p in points)
    out: list[ConfusionPoint] = []
    for t, s in times_scores:
        lo = t - window_sec
        acc = [sc for ts, sc in times_scores if lo < ts <= t]
        mean = sum(acc) / len(acc) if acc else s
        out.append(ConfusionPoint(t_sec=t, score=mean))
    return out
