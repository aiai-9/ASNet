#!/usr/bin/env bash
set -euo pipefail

# ================== CONFIG ==================
# Default commit message if you don't pass one:
DEFAULT_MSG_PREFIX="autosync"
# ===========================================

# Go to repo root (git_script/..)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "📂 Repo: $REPO_ROOT"

# Make sure this is a git repo
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "❌ Not inside a git repository."
  exit 1
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "👉 Current branch: $BRANCH"
echo

# Commit message (arg1 or auto timestamp)
TIMESTAMP="$(date +"%Y-%m-%d %H:%M:%S")"
COMMIT_MSG="${1:-$DEFAULT_MSG_PREFIX: $TIMESTAMP}"

# 1) Show short status
echo "🔍 git status --short:"
git status --short
echo

# 2) If there are any changes (tracked or untracked), stage + commit them
HAS_CHANGES=false
if ! git diff --quiet || ! git diff --cached --quiet; then
  HAS_CHANGES=true
fi
if [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
  HAS_CHANGES=true
fi

if [[ "$HAS_CHANGES" = true ]]; then
  echo "➕ Staging all changes (git add .)..."
  git add .

  if git diff --cached --quiet; then
    echo "ℹ️ Nothing staged after git add . (maybe only ignored files)."
  else
    echo "📝 Committing with message: \"$COMMIT_MSG\""
    git commit -m "$COMMIT_MSG"
  fi
else
  echo "✅ No local changes to commit."
fi

echo

# 3) Pull latest from remote (with rebase)
echo "⬇️  Pulling latest changes (git pull --rebase)..."
if ! git pull --rebase; then
  echo
  echo "❌ git pull --rebase failed (likely merge conflicts)."
  echo "   Please resolve conflicts manually, then run:"
  echo "   - git add <fixed files>"
  echo "   - git rebase --continue"
  echo "   After that, re-run this script."
  exit 1
fi
echo "✅ Pull successful."
echo

# 4) Check if we are ahead and need to push
AHEAD=false
if git status -sb | grep -q '\[ahead '; then
  AHEAD=true
fi

if [[ "$AHEAD" = true ]]; then
  echo "⬆️  Pushing to remote (git push)..."
  if ! git push; then
    echo
    echo "❌ git push failed (network/GitHub issue?)."
    echo "   Try again later with:"
    echo "   - git push"
    exit 1
  fi
  echo "✅ Push complete."
else
  echo "✅ Nothing to push (already in sync with remote)."
fi

echo
echo "🎉 git_sync.sh finished successfully."


# ======================================== Make script executable ========================================

# chmod +x git_script/git_sync.sh

# ./git_script/git_sync.sh "my commit message"