"""Shared data structures for timelines and reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class FrameSample:
    t_sec: float
    path: str


@dataclass
class ConfusionPoint:
    t_sec: float
    score: float


@dataclass
class TranscriptSegment:
    t_start: float
    t_end: float
    text: str


Quality = Literal["high", "low", "unknown"]
TalkMove = str


@dataclass
class AnnotatedUtterance:
    t_start: float
    t_end: float
    text: str
    talk_move: TalkMove
    strategy_quality: Quality
    uptake_score: float | None = None
    notes: str = ""


@dataclass
class StrategyBinSummary:
    t_start: float
    t_end: float
    dominant_quality: Quality
    low_quality: bool
    high_pressure: bool
    talk_moves: list[str] = field(default_factory=list)
    transcript_excerpt: str = ""


@dataclass
class AlignmentBin:
    t_start: float
    t_end: float
    mean_confusion: float
    strategy: StrategyBinSummary


@dataclass
class FrictionWindow:
    """One candidate friction interval for multimodal VLM fusion."""

    t_start_sec: float
    t_end_sec: float
    confusion_summary: str
    strategy_summary: str
    transcript_excerpt: str
    frame_paths: list[str | Path]


@dataclass
class FrictionCandidate:
    t_start: float
    t_end: float
    mean_confusion: float
    confusion_z: float
    strategy: StrategyBinSummary
    frame_paths: list[str]


@dataclass
class FusionResult:
    t_start: float
    t_end: float
    friction: bool | None
    rationale: str
    alternative_strategy: str
    raw_model_output: dict | None = None


@dataclass
class TeacherFrictionReport:
    video_path: str
    bins: list[AlignmentBin]
    candidates: list[FrictionCandidate]
    fusion: list[FusionResult]
    work_dir: str
