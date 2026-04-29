#!/bin/bash
#SBATCH --job-name=friction-pipeline
#SBATCH --output=/scratch/jw310/comp646/runs/logs/pipeline_%A_%a.log
#SBATCH --error=/scratch/jw310/comp646/runs/logs/pipeline_%A_%a.err
#SBATCH --partition=commons
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-53%8

set -e

REPO="/scratch/jw310/comp646"
SCRATCH="/scratch/jw310/comp646"
CATALOG="$REPO/data/catalog.csv"

# Make log dir
mkdir -p "$SCRATCH/runs/logs"
mkdir -p "$SCRATCH/runs/video_pipeline"

cd "$REPO"
export PYTHONPATH="$REPO/src:$PYTHONPATH"

# Install deps if needed
pip install --quiet transformers torch opencv-python-headless faster-whisper deepface imageio-ffmpeg tf-keras 2>/dev/null || true

# Read all rows from catalog (skip header)
mapfile -t ROWS < <(tail -n +2 "$CATALOG")
ROW="${ROWS[$SLURM_ARRAY_TASK_ID]}"
if [ -z "$ROW" ]; then
    echo "No row for task $SLURM_ARRAY_TASK_ID"
    exit 0
fi

YOUTUBE_ID=$(echo "$ROW" | cut -d',' -f1)
LESSON_ID=$(echo "$ROW"  | cut -d',' -f2)
SUBJECT=$(echo "$ROW"    | cut -d',' -f3)

# Find video file
if [ "$SUBJECT" = "math" ]; then
    VIDEO="$SCRATCH/data/math/videos/TIMSS_${YOUTUBE_ID}.mp4"
else
    VIDEO="$SCRATCH/data/science/videos/TIMSS_${YOUTUBE_ID}.mp4"
fi

if [ ! -f "$VIDEO" ]; then
    echo "[$LESSON_ID] Video not found: $VIDEO"
    exit 1
fi

OUT_DIR="$SCRATCH/runs/video_pipeline/$LESSON_ID"
mkdir -p "$OUT_DIR"

echo "[$LESSON_ID] Starting: $VIDEO"
echo "[$LESSON_ID] Output:   $OUT_DIR"
echo "[$LESSON_ID] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"

python3 - <<PYEOF
import sys, os
sys.path.insert(0, "/scratch/jw310/comp646/src")
from pathlib import Path
from config import PipelineConfig
from pipeline.run import run_pipeline

cfg = PipelineConfig(
    vision_sample_fps=2.0,
    confusion_sliding_window_sec=30.0,
    alignment_grid_sec=30.0,
    confusion_z_threshold=1.0,
    whisper_model_size="small",
    whisper_device="cuda",
    whisper_compute_type="float16",
    work_dir="$OUT_DIR",
    skip_fusion=True,
    flag_unknown_strategy=True,
    max_duration_sec=None,
)

report = run_pipeline("$VIDEO", cfg)

print(f"[$LESSON_ID] DONE — bins={len(report.bins)}, candidates={len(report.candidates)}")
for c in report.candidates:
    print(f"  {c.t_start:.0f}-{c.t_end:.0f}s  confusion={c.mean_confusion:.3f} z={c.confusion_z:.2f}  quality={c.strategy.dominant_quality}")
PYEOF
