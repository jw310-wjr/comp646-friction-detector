#!/bin/bash
#SBATCH --job-name=friction-qwen
#SBATCH --output=runs/gpu_fusion_%j.log
#SBATCH --error=runs/gpu_fusion_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
# --mail-user=YOUR_EMAIL@rice.edu   # uncomment and fill in

set -e

REPO="/scratch/jw310/comp646"
VIDEO="$REPO/data/math/videos/TIMSS_RUBHh0bPScw.mp4"
WORK_DIR="$REPO/runs/gpu_au1_full"

cd "$REPO"

# --- environment setup ---
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --quiet -r requirements.txt
fi

# Check that the video exists
if [ ! -f "$VIDEO" ]; then
    echo "ERROR: video not found at $VIDEO"
    echo "Copy the AU1 .mp4 file there first."
    exit 1
fi

echo "=== GPU Fusion Run: AU1 full lesson ==="
echo "Video : $VIDEO"
echo "Output: $WORK_DIR"
echo "GPU   : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "========================================"

python3 - <<'PYEOF'
import sys, os
sys.path.insert(0, os.path.join(os.environ["REPO"], "src"))
from pathlib import Path
from config import PipelineConfig
from pipeline.run import run_pipeline

cfg = PipelineConfig(
    vision_sample_fps=2.0,
    confusion_sliding_window_sec=30.0,
    alignment_grid_sec=30.0,
    confusion_z_threshold=0.3,
    whisper_model_size="small",   # better quality than "base"
    whisper_device="cuda",
    whisper_compute_type="float16",
    qwen_model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    qwen_max_new_tokens=512,
    work_dir=os.environ["WORK_DIR"],
    skip_fusion=False,            # VLM fusion ON
    max_fusion_windows=20,
    frames_per_candidate=4,
    flag_unknown_strategy=True,
    max_duration_sec=None,        # full lesson
)

report = run_pipeline(os.environ["VIDEO"], cfg)

print(f"\n=== DONE ===")
print(f"Bins          : {len(report.bins)}")
print(f"Candidates    : {len(report.candidates)}")
print(f"Fusion results: {len(report.fusion)}")
confirmed = [f for f in report.fusion if f.friction is True]
print(f"Friction=True : {len(confirmed)}")
for f in report.fusion:
    flag = "FRICTION" if f.friction else ("skip" if f.friction is None else "ok")
    print(f"  [{flag:8s}] {f.t_start:.0f}s-{f.t_end:.0f}s  {f.rationale[:80]}")
PYEOF
