"""Sample video frames and build a confusion timeline via DeepFace emotion (proposal §2)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from schemas import ConfusionPoint, FrameSample

try:
    from deepface import DeepFace
except ImportError as e:  # pragma: no cover
    DeepFace = None
    _DEEPFACE_ERR = e
else:
    _DEEPFACE_ERR = None


# Map DeepFace emotions to a simple confusion proxy (0 = calm, 1 = strong confusion / negative affect).
_EMOTION_CONFUSION_WEIGHT: dict[str, float] = {
    "neutral": 0.0,
    "happy": 0.0,
    "surprise": 0.25,
    "fear": 0.9,
    "sad": 0.85,
    "angry": 0.45,
    "disgust": 0.5,
}


def get_video_duration_sec(video_path: str | Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return float(n / fps) if fps > 0 else 0.0


def extract_frames_uniform_fps(
    video_path: str | Path,
    target_fps: float,
    out_dir: str | Path,
) -> list[FrameSample]:
    """Uniformly sample frames at ``target_fps`` and write JPEGs to ``out_dir``."""
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / src_fps if src_fps > 0 else 0.0
    if duration <= 0:
        duration = 1.0
    step = src_fps / max(target_fps, 1e-6)
    samples: list[FrameSample] = []
    idx = 0.0
    frame_id = 0
    pbar = tqdm(total=max(1, int(duration * target_fps)), desc="Extracting frames", unit="fr")
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(idx)))
        ok, frame = cap.read()
        if not ok:
            break
        t_sec = idx / src_fps
        name = f"f_{frame_id:06d}_{t_sec:.3f}s.jpg"
        path = out_dir / name
        cv2.imwrite(str(path), frame)
        samples.append(FrameSample(t_sec=float(t_sec), path=str(path.resolve())))
        frame_id += 1
        idx += step
        pbar.update(1)
        if idx >= frame_count:
            break
    pbar.close()
    cap.release()
    return samples


def _dominant_emotion_score(frame_path: str, enforce_detection: bool) -> float:
    if DeepFace is None:
        raise RuntimeError(
            "deepface is not installed or failed to import"
        ) from _DEEPFACE_ERR
    objs = DeepFace.analyze(
        img_path=frame_path,
        actions=["emotion"],
        enforce_detection=enforce_detection,
        silent=True,
    )
    if isinstance(objs, dict):
        objs = [objs]
    scores = []
    for o in objs:
        emo = o.get("dominant_emotion") or max(o["emotion"], key=o["emotion"].get)  # type: ignore[arg-type]
        emo_s = str(emo).lower()
        conf = float(_EMOTION_CONFUSION_WEIGHT.get(emo_s, 0.35))
        scores.append(conf)
    return float(np.mean(scores)) if scores else 0.0


def build_confusion_timeline(
    frames: list[FrameSample],
    enforce_detection: bool = False,
) -> list[ConfusionPoint]:
    """Per-frame confusion proxy, then caller applies 30s sliding average (proposal)."""
    pts: list[ConfusionPoint] = []
    for fr in tqdm(frames, desc="DeepFace emotion"):
        try:
            s = _dominant_emotion_score(fr.path, enforce_detection=enforce_detection)
        except Exception:
            s = 0.0
        pts.append(ConfusionPoint(t_sec=fr.t_sec, score=s))
    return pts
