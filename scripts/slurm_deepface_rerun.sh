#!/bin/bash
#SBATCH --job-name=friction-deepface
#SBATCH --output=/scratch/jw310/comp646/runs/logs/deepface_%A_%a.log
#SBATCH --error=/scratch/jw310/comp646/runs/logs/deepface_%A_%a.err
#SBATCH --partition=commons
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-53%10

set -e

REPO="/scratch/jw310/comp646"
mkdir -p "$REPO/runs/logs"

# Fix DeepFace tf-keras dependency
pip install --quiet --user tf-keras 2>/dev/null || true

export PYTHONPATH="$REPO/src:$PYTHONPATH"

# Get lesson ID from catalog (skip header)
mapfile -t LESSONS < <(tail -n +2 "$REPO/data/catalog.csv" | cut -d',' -f2)
LESSON="${LESSONS[$SLURM_ARRAY_TASK_ID]}"

if [ -z "$LESSON" ]; then
    echo "No lesson for task $SLURM_ARRAY_TASK_ID"
    exit 0
fi

echo "[$LESSON] Starting DeepFace + fusion  (GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo cpu))"

python3 "$REPO/scripts/rerun_deepface_fusion.py" --lesson "$LESSON"

echo "[$LESSON] Done"
