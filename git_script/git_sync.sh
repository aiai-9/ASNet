#!/usr/bin/env bash
set -euo pipefail

# ================== CONFIG ==================
# Default commit message prefix if you don't pass one:
DEFAULT_MSG_PREFIX="autosync"
# ===========================================

# Go to repo root (git_script/..)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "📂 Repo: $REPO_ROOT"

# Ensure we are inside a git repo
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "❌ Not inside a git repository."
  exit 1
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "👉 Current branch: $BRANCH"
echo

# Commit message:
# - If you pass arguments, use them as the message
# - Otherwise, use "autosync: <timestamp>"
TIMESTAMP="$(date +"%Y-%m-%d %H:%M:%S")"
if [[ $# -gt 0 ]]; then
  COMMIT_MSG="$*"
else
  COMMIT_MSG="$DEFAULT_MSG_PREFIX: $TIMESTAMP"
fi

# 1) Show short git status
echo "🔍 git status --short:"
git status --short || true
echo

# 2) Detect local changes (tracked or untracked)
HAS_CHANGES=false
if ! git diff --quiet || ! git diff --cached --quiet; then
  HAS_CHANGES=true
fi
if [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
  HAS_CHANGES=true
fi

# 3) If there are changes, stage and commit
if [[ "$HAS_CHANGES" == true ]]; then
  echo "➕ Staging all changes (git add .)..."
  git add .

  # If still nothing staged, everything was ignored
  if git diff --cached --quiet; then
    echo "ℹ️ Only ignored files changed; nothing to commit."
  else
    echo "📝 Committing with message: \"$COMMIT_MSG\""
    git commit -m "$COMMIT_MSG"
  fi
else
  echo "✅ No local changes to commit."
fi

echo

# 4) Check if there is an upstream (remote tracking) branch
HAS_UPSTREAM=true
if ! git rev-parse --abbrev-ref --symbolic-full-name @{u} >/dev/null 2>&1; then
  HAS_UPSTREAM=false
fi

if [[ "$HAS_UPSTREAM" == false ]]; then
  echo "⚠️  No upstream branch configured for '$BRANCH'."
  echo "    First run:"
  echo "      git push -u origin $BRANCH"
  echo "    After that, re-run this script."
  exit 0
fi

# 5) Pull latest from remote (with rebase)
echo "⬇️  Pulling latest changes (git pull --rebase)..."
if ! git pull --rebase; then
  echo
  echo "❌ git pull --rebase failed (likely conflicts or rebase in progress)."
  echo "   Please resolve manually, then run:"
  echo "     git status"
  echo "     # fix conflicts"
  echo "     git add <fixed files>"
  echo "     git rebase --continue"
  echo "   After that, re-run this script."
  exit 1
fi
echo "✅ Pull successful."
echo

# 6) Check if we are ahead of remote and need to push
AHEAD=false
if git status -sb | grep -q '\[ahead '; then
  AHEAD=true
fi

if [[ "$AHEAD" == true ]]; then
  echo "⬆️  Pushing to remote (git push)..."
  if ! git push; then
    echo
    echo "❌ git push failed."
    echo "   Common causes:"
    echo "   - Missing / invalid GitHub credentials"
    echo "   - Network issues"
    echo
    echo "   To fix credentials (once per machine):"
    echo "     git config --global credential.helper store"
    echo "     git push        # enter GitHub username + PAT when asked"
    echo "   Then re-run:"
    echo "     ./git_script/git_sync.sh \"<msg>\""
    exit 1
  fi
  echo "✅ Push complete."
else
  echo "✅ Nothing to push (local branch is in sync with remote)."
fi

echo
echo "🎉 git_sync.sh finished successfully."

aiai-9

# ======================================== Make script executable ========================================

# chmod +x git_script/git_sync.sh

# ./git_script/git_sync.sh "my commit message"

# ./git_script/git_sync.sh "fix: calibration bug in ASNet"