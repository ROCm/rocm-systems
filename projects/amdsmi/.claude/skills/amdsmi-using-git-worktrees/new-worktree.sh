#!/usr/bin/env bash
# Create an amd-smi worktree following the rocm-systems-<name> sibling convention.
# Derives the parent and base from git; no hard-coded users or paths, so any
# rocm-systems checkout can use it.
#
# Usage:
#   new-worktree.sh <name> [-b branch] [--base origin/develop]   # new/existing branch off base
#   new-worktree.sh -p <PR#> [-b branch]                          # names it rocm-systems-pr<PR#>
#
# Examples:
#   new-worktree.sh -p 10091
#   new-worktree.sh apu-metric-schema -b users/me/apu-metric-schema
#
# Prints a `cd <path>` line on stdout; run `eval "$(new-worktree.sh ...)"` to jump in.
set -uo pipefail

BASE="origin/develop"
PR=""; NAME=""; BRANCH=""
need_val() { [[ $# -ge 2 ]] || { echo "error: $1 needs a value" >&2; exit 2; }; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--pr)     need_val "$@"; PR="$2"; shift 2;;
    -b|--branch) need_val "$@"; BRANCH="$2"; shift 2;;
    --base)      need_val "$@"; BASE="$2"; shift 2;;
    -h|--help)   tail -n +2 "$0" | grep '^#' | sed 's/^# \{0,1\}//'; exit 0;;
    -*) echo "unknown flag: $1" >&2; exit 2;;
    *)  NAME="$1"; shift;;
  esac
done

git rev-parse --is-inside-work-tree >/dev/null 2>&1 \
  || { echo "error: not inside a git repo" >&2; exit 2; }

# Derive the *main* checkout even when invoked from inside a linked worktree.
MAIN_CHECKOUT=$(git -C "$(git rev-parse --git-common-dir)/.." rev-parse --show-toplevel)
PARENT=$(dirname "$MAIN_CHECKOUT")

# Worktree directory name from the tiered convention.
if [[ -n "$PR" ]]; then
  DIRNAME="rocm-systems-pr${PR}"
elif [[ -n "$NAME" ]]; then
  DIRNAME="rocm-systems-${NAME}"
else
  echo "error: pass a <name> or -p <PR#>" >&2; exit 2
fi
WORKTREE="${PARENT}/${DIRNAME}"
[[ -e "$WORKTREE" ]] && { echo "error: $WORKTREE already exists" >&2; exit 1; }

cd "$MAIN_CHECKOUT"

if [[ -n "$PR" ]]; then
  BRANCH="${BRANCH:-pr-${PR}}"
  git fetch origin "pull/${PR}/head:${BRANCH}" 2>/dev/null \
    || git fetch origin "${BRANCH}:${BRANCH}"
  git worktree add "$WORKTREE" "$BRANCH"
elif [[ -n "$BRANCH" ]]; then
  git fetch origin "${BASE#origin/}" 2>/dev/null || true
  git worktree add "$WORKTREE" -b "$BRANCH" "$BASE"
else
  git worktree add "$WORKTREE" "$BASE"
fi

# amd-smi work happens under projects/amdsmi in the monorepo layout.
DEST="$WORKTREE"
[[ -d "$WORKTREE/projects/amdsmi" ]] && DEST="$WORKTREE/projects/amdsmi"

echo "created: $WORKTREE" >&2
echo "branch:  $(git -C "$WORKTREE" branch --show-current 2>/dev/null || echo '(detached)')" >&2
echo "cd $DEST"
