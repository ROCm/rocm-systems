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
# Most hooks here are advisory: they always pass, so a green job is exactly
# what clang-format and pylint produce whether they found nothing or found
# plenty. Left in the log, their output lives inside a collapsed section of a
# passing step, which is to say nobody reads it.
#
# -f before -s, as in ci-clang-tidy.sh: every way the run can die before
# pre-commit speaks leaves no log at all, and testing only for emptiness would
# report a run that never happened as a clean one.
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

# Append to the job summary under CI, print it when run by hand. Not a
# `>> "${GITHUB_STEP_SUMMARY:-/dev/stdout}"` default: reopening /dev/stdout
# fails outright whenever stdout is something that cannot be reopened, such as
# a pipe or a socket, and from an EXIT trap that failure would turn a clean run
# red for no reason anyone could act on.
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
# unreachable hook repository still reports that where someone will see it. A
# trap handler does not change the exit status unless it exits itself, so this
# neither masks that failure nor disturbs the status of the run below.
trap emit_summary EXIT

# Tee rather than redirect: the step log stays the place to look when the job
# fails, and the summary needs the same text. The log lives outside the
# checkout so the hooks never see their own transcript. pipefail so the status
# of the run survives the pipe, since tee always succeeds.
set -o pipefail
pre-commit run \
  --config "$CONFIG" \
  --from-ref "$BASE_REF" \
  --to-ref "$HEAD_REF" \
  2>&1 | tee "$LOG"
