#!/usr/bin/env bash
#
# ci-pre-commit.sh — Drive the ordinary-stage pre-commit hooks in CI.
#
# The sibling of ci-clang-tidy.sh: that one runs the manual-stage clang-tidy
# hook, this one runs everything else, over the files a pull request changed.
#
# Simulate what CI does locally by running this from the repository root:
#
#   projects/rccl/maint/ci-pre-commit.sh origin/develop HEAD
#
# Usage:
#   ci-pre-commit.sh [BASE_REF HEAD_REF]
#
# With no arguments the range defaults to origin/develop..HEAD. Unlike
# ci-clang-tidy.sh this neither installs packages nor adds git remotes, so it
# is safe to run in your normal clone.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

BASE_REF="${1:-origin/develop}"
HEAD_REF="${2:-HEAD}"

CONFIG="projects/rccl/.pre-commit-config.yaml"
LOG="${RUNNER_TEMP:-/tmp}/rccl-pre-commit.txt"

# ---------------------------------------------------------------------------
# Summarise the run for the job summary
# ---------------------------------------------------------------------------
# shellcheck disable=SC2329  # invoked from the EXIT trap armed below
summarise() {
  echo "## pre-commit"
  echo
  if [[ ! -f "$LOG" || ! -s "$LOG" ]]; then
    echo ":warning: pre-commit produced no output, so nothing was checked."
    echo "See the \`Run pre-commit\` step log for the error."
  else
    echo '```'
    cat "$LOG"
    echo '```'
  fi
}

# shellcheck disable=SC2329  # invoked from the EXIT trap armed below
emit_summary() {
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    summarise >> "$GITHUB_STEP_SUMMARY"
  else
    summarise
  fi
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if ! command -v pre-commit &>/dev/null; then
  echo "ERROR: pre-commit not found. Install it with 'pip install pre-commit'" >&2
  echo "       (or 'uv tool install pre-commit')." >&2
  exit 2
fi

# Armed before the run, so a pre-commit that dies on a bad config or an
# unreachable hook repository still reports that where someone will see it.
trap emit_summary EXIT

set -o pipefail
pre-commit run \
  --config "$CONFIG" \
  --from-ref "$BASE_REF" \
  --to-ref "$HEAD_REF" \
  2>&1 | tee "$LOG"
