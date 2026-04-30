#!/usr/bin/env bash
# Batch Claude fusion across all 53 TIMSS lessons.
# Usage:
#   export ANTHROPIC_API_KEY=sk-ant-...
#   bash scripts/run_all_fusion.sh
#
# Results land in runs/claude_fusion/<lesson_id>/fusion_heuristic.json
# Run make_figures.py afterwards to generate the aggregate fusion figure.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRANSCRIPT_RUNS="$ROOT/runs/transcripts"

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ERROR: ANTHROPIC_API_KEY is not set. Export it before running this script."
  exit 1
fi

LESSONS=$(ls "$TRANSCRIPT_RUNS" | grep -v aggregate_results.json)
TOTAL=$(echo "$LESSONS" | wc -l)
DONE=0
SKIPPED=0

echo "Running Claude fusion on $TOTAL lessons..."
echo ""

for lesson in $LESSONS; do
  OUT="$ROOT/runs/claude_fusion/$lesson/fusion_heuristic.json"
  if [[ -f "$OUT" ]]; then
    echo "  [SKIP] $lesson — already done"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi
  echo "  [$((DONE+SKIPPED+1))/$TOTAL] $lesson"
  python "$ROOT/scripts/run_claude_fusion.py" --lesson "$lesson" --annotator heuristic
  DONE=$((DONE + 1))
done

echo ""
echo "Finished: $DONE processed, $SKIPPED already done."
echo "Regenerating figures..."
python "$ROOT/scripts/make_figures.py"
