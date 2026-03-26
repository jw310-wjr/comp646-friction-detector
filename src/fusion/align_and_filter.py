"""Align vision + language on a shared grid; heuristic pre-filter (proposal §2)."""

from __future__ import annotations

import statistics

from schemas import (
    AlignmentBin,
    AnnotatedUtterance,
    ConfusionPoint,
    FrictionCandidate,
    Quality,
    StrategyBinSummary,
)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _summarize_bin_utterances(
    utts: list[AnnotatedUtterance],
    t0: float,
    t1: float,
) -> StrategyBinSummary:
    sel = [u for u in utts if _overlap(u.t_start, u.t_end, t0, t1) > 0]
    if not sel:
        return StrategyBinSummary(
            t_start=t0,
            t_end=t1,
            dominant_quality="unknown",
            low_quality=False,
            high_pressure=False,
            talk_moves=[],
            transcript_excerpt="",
        )
    text = " ".join(u.text for u in sel)
    qualities: list[Quality] = [u.strategy_quality for u in sel]
    low = any(q == "low" for q in qualities)
    high_pressure = any("Pressing for Accuracy" in u.talk_move for u in sel)
    dom: Quality
    if low:
        dom = "low"
    elif all(q == "high" for q in qualities):
        dom = "high"
    else:
        dom = "unknown"
    moves = list({u.talk_move for u in sel})
    excerpt = text[:1200]
    return StrategyBinSummary(
        t_start=t0,
        t_end=t1,
        dominant_quality=dom,
        low_quality=low,
        high_pressure=high_pressure,
        talk_moves=moves,
        transcript_excerpt=excerpt,
    )


def _mean_confusion_in_bin(points: list[ConfusionPoint], t0: float, t1: float) -> float:
    vals = [p.score for p in points if t0 <= p.t_sec < t1]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def build_alignment_bins(
    smoothed_confusion: list[ConfusionPoint],
    utterances: list[AnnotatedUtterance],
    grid_sec: float,
    video_duration_sec: float,
) -> list[AlignmentBin]:
    bins: list[AlignmentBin] = []
    t = 0.0
    while t < video_duration_sec + 1e-6:
        t0, t1 = t, t + grid_sec
        mean_c = _mean_confusion_in_bin(smoothed_confusion, t0, t1)
        strat = _summarize_bin_utterances(utterances, t0, t1)
        bins.append(
            AlignmentBin(
                t_start=t0,
                t_end=t1,
                mean_confusion=mean_c,
                strategy=strat,
            )
        )
        t = t1
    return bins


def heuristic_candidates(
    bins: list[AlignmentBin],
    confusion_z: float,
    frame_index: list[tuple[float, str]],
) -> tuple[list[FrictionCandidate], float, float]:
    """
    Flag bins where confusion > μ + z·σ and (low-quality strategy or high-pressure move).
    ``frame_index`` is sorted (t_sec, path) for all extracted frames.
    """
    scores = [b.mean_confusion for b in bins]
    if not scores:
        mu, sigma = 0.0, 0.0
    elif len(scores) < 2:
        mu, sigma = float(scores[0]), 0.0
    else:
        mu = float(statistics.mean(scores))
        sigma = float(statistics.pstdev(scores)) or 0.0

    cands: list[FrictionCandidate] = []
    for b in bins:
        z = (b.mean_confusion - mu) / sigma if sigma > 1e-9 else 0.0
        risky = b.strategy.low_quality or b.strategy.high_pressure
        if b.mean_confusion > mu + confusion_z * sigma and risky:
            fps = [p for t, p in frame_index if b.t_start <= t < b.t_end]
            if not fps and frame_index:
                mid = (b.t_start + b.t_end) / 2
                fps = [min(frame_index, key=lambda x: abs(x[0] - mid))[1]]
            cands.append(
                FrictionCandidate(
                    t_start=b.t_start,
                    t_end=b.t_end,
                    mean_confusion=b.mean_confusion,
                    confusion_z=z,
                    strategy=b.strategy,
                    frame_paths=fps,
                )
            )
    return cands, mu, sigma
