#!/bin/bash
# Stand up a worktree carrying the AIRUNTIME-28 change, ready for clr_build.sh.
#
# The change lives as a commit on users/victzhan/AIRUNTIME-28-nt-blit; the patch
# file is derived from it by clr_patch.sh. This script prefers the branch and
# falls back to applying the patch on a detached base, so it works both where the
# branch exists and from a fresh clone that only has the patch.
set -euo pipefail
cd "$(dirname "$0")"
PATCH=$(pwd)/airuntime28-nt-blit.patch

BRANCH=users/victzhan/AIRUNTIME-28-nt-blit
# Fallback base for the patch path, read from the patch's own header so it cannot
# go stale when the branch is rebased. The measurements in REPORT.md were taken
# at 563095dbca, before the branch was moved forward onto develop.
BASE=$(sed -n 's/^# base:[[:space:]]*//p' "$PATCH" | head -1)
BASE=${BASE:-origin/develop}
WT=~/airuntime28-clr
UPSTREAM=${UPSTREAM:-~/rocm-systems}

cd "$UPSTREAM"
if [ -d "$WT" ]; then
  echo "removing previous worktree"
  git worktree remove --force "$WT" 2>/dev/null || rm -rf "$WT"
  git worktree prune
fi

if git rev-parse --verify --quiet "$BRANCH" >/dev/null; then
  echo "=== using existing branch $BRANCH ==="
  git worktree add "$WT" "$BRANCH" 2>&1 | tail -2
  cd "$WT"
  git log --oneline -1
else
  echo "=== branch $BRANCH not found; applying the patch at $BASE ==="
  git worktree add --detach "$WT" "$BASE" 2>&1 | tail -2
  cd "$WT"
  git log --oneline -1
  git apply --stat "$PATCH"
  git apply "$PATCH"
  git diff --stat
fi
echo "CLR_SETUP_OK"
