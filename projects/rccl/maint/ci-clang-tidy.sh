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
#     projects/rccl/maint/ci-clang-tidy.sh
#
# To run clang-tidy in your own work tree, use maint/run-clang-tidy.sh directly.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

FINDINGS="projects/rccl/build/clang-tidy/clang-tidy-findings.txt"

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
summarise() {
  echo "## clang-tidy (advisory)"
  echo
  # -f before -s: every way the analysis can die leaves no findings file at
  # all. Testing only for emptiness would report a run that never happened as
  # a clean one.
  if [[ ! -f "$FINDINGS" ]]; then
    echo ":warning: clang-tidy did not complete, so nothing was analysed."
    echo "See the \`Run clang-tidy\` step log for the error."
  elif [[ ! -s "$FINDINGS" ]]; then
    echo "No findings in RCCL-owned files."
  else
    echo "$(wc -l < "$FINDINGS") finding(s) in RCCL-owned files:"
    echo
    echo '```'
    cat "$FINDINGS"
    echo '```'
  fi
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
install_build_deps
fetch_nccl_ref
install_pre_commit

export PATH="/opt/rocm/bin:${VENV_DIR}/bin:${PATH}"

STATUS=0
pre-commit run \
  --config projects/rccl/.pre-commit-config.yaml \
  --hook-stage manual \
  clang-tidy || STATUS=$?

summarise >> "${GITHUB_STEP_SUMMARY:-/dev/stdout}"

exit "$STATUS"
