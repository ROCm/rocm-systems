#!/usr/bin/env bash
#
# ci-clang-tidy.sh — Drive the advisory clang-tidy check in CI.
#
# Simulate what CI does locally by running this from the repository
# root of a checkout that contains projects/rccl (it adds remotes
# so don't use it in your normal clone):
#
#   podman run --rm -it -v "$PWD:/src" -w /src \
#     rocm/dev-ubuntu-26.04:10.1.0-full \
#     projects/rccl/maint/ci-clang-tidy.sh origin/develop HEAD
#
# Usage:
#   ci-clang-tidy.sh [BASE_REF HEAD_REF]
#
# Only the C/C++ files changed between BASE_REF and HEAD_REF are analysed
# (defaults to origin/develop..HEAD). Findings reachable only through a changed
# header are not reported.
#
# To run clang-tidy in your own work tree, use maint/run-clang-tidy.sh directly.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

BASE_REF="${1:-origin/develop}"
HEAD_REF="${2:-HEAD}"

FINDINGS="projects/rccl/build/clang-tidy/clang-tidy-findings.txt"
NOT_ANALYSED="projects/rccl/build/clang-tidy/clang-tidy-not-analysed.txt"

LOG="${RUNNER_TEMP:-/tmp}/rccl-clang-tidy-precommit.txt"

VENV_DIR="${RUNNER_TEMP:-/tmp}/rccl-clang-tidy-venv"

# ---------------------------------------------------------------------------
# Install what configure + hipify need
# ---------------------------------------------------------------------------
install_build_deps() {
  apt-get update
  apt-get install --assume-yes --no-install-recommends \
    build-essential \
    cmake \
    git \
    perl \
    python3 \
    python3-venv \
    rocm-cmake
}

# ---------------------------------------------------------------------------
# Fetch the NCCL ref used to tell RCCL-owned files from upstream ones
# ---------------------------------------------------------------------------
fetch_nccl_ref() {
  git config --global --add safe.directory "$REPO_ROOT"
  if ! git remote get-url nccl >/dev/null 2>&1; then
    git remote add nccl https://github.com/NVIDIA/nccl.git
    git remote set-url --push nccl no_push
  fi
  git fetch --no-tags --depth=1 --filter=blob:none nccl master
}

# ---------------------------------------------------------------------------
# Install pre-commit
# ---------------------------------------------------------------------------
install_pre_commit() {
  python3 -m venv "$VENV_DIR"
  "${VENV_DIR}/bin/pip" install --quiet pre-commit
}

# ---------------------------------------------------------------------------
# Summarise the findings for the job summary
# ---------------------------------------------------------------------------
# shellcheck disable=SC2329  # invoked from the EXIT trap armed below
summarise() {
  echo "## clang-tidy (advisory)"
  echo
  # -f before -s: every way the analysis can die leaves no findings file at
  # all. Testing only for emptiness would report a run that never happened as
  # a clean one.
  # pre-commit skips the hook, without running it, when the change touches no
  # C/C++ file under src/; that is not a failure.
  if [[ ! -f "$FINDINGS" ]] && grep -q 'no files to check' "$LOG" 2>/dev/null; then
    echo "No C/C++ files under src/ changed, so there was nothing to analyse."
  # A successful run that leaves no findings file analysed nothing: every
  # changed file is missing from the compile database (listed below).
  elif [[ ! -f "$FINDINGS" && -s "$NOT_ANALYSED" && "${STATUS:-1}" -eq 0 ]]; then
    echo "Nothing was analysed: none of the changed files is in the compile database."
  elif [[ ! -f "$FINDINGS" ]]; then
    echo ":warning: clang-tidy did not complete, so nothing was analysed."
    echo "See the \`Run clang-tidy\` step log for the error."
  elif [[ ! -s "$FINDINGS" ]]; then
    echo "No findings in RCCL-owned files among the files this change touched."
  else
    echo "$(wc -l < "$FINDINGS") finding(s) in RCCL-owned files among the files this change touched:"
    echo
    echo '```'
    cat "$FINDINGS"
    echo '```'
  fi
  if [[ -s "$NOT_ANALYSED" ]]; then
    echo
    echo "Not analysed (not in the compile database):"
    echo
    # shellcheck disable=SC2016  # literal backticks for Markdown
    sed 's|^|- `|; s|$|`|' "$NOT_ANALYSED"
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
trap emit_summary EXIT

install_build_deps
fetch_nccl_ref
install_pre_commit

export PATH="/opt/rocm/bin:${VENV_DIR}/bin:${PATH}"

# A leftover file from an earlier run would be mistaken for this run's result.
rm -f "$FINDINGS" "$NOT_ANALYSED"

STATUS=0
set -o pipefail
pre-commit run \
  --config projects/rccl/.pre-commit-config.yaml \
  --from-ref "$BASE_REF" \
  --to-ref "$HEAD_REF" \
  --hook-stage manual \
  clang-tidy 2>&1 | tee "$LOG" || STATUS=$?

exit "$STATUS"
