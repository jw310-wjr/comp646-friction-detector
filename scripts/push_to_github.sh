#!/usr/bin/env bash
# Create GitHub repo and push (requires: brew install gh && gh auth login)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
REPO_NAME="${1:-comp646-friction-detector}"
VISIBILITY="${2:-public}"

if ! command -v gh >/dev/null 2>&1; then
  echo "Install GitHub CLI: brew install gh"
  exit 1
fi
if ! gh auth status >/dev/null 2>&1; then
  echo "Not logged in. Run: gh auth login"
  exit 1
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git init
  git branch -M main
fi
git add -A
if git diff --cached --quiet; then
  echo "Nothing to commit."
else
  git commit -m "Initial commit: multimodal pedagogical friction detector pipeline"
fi

if git remote get-url origin >/dev/null 2>&1; then
  echo "Remote origin already set. Pushing..."
  git push -u origin main
else
  if [ "$VISIBILITY" = "private" ]; then
    gh repo create "$REPO_NAME" --private --source=. --remote=origin --push
  else
    gh repo create "$REPO_NAME" --public --source=. --remote=origin --push
  fi
fi

echo "Done: https://github.com/$(gh api user -q .login)/$REPO_NAME"
