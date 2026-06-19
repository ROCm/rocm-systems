#!/bin/bash
###############################################################################
# Performance Regression Comparison
#
# Compares one or more build variants against a baseline. By default compares
# the current branch against its merge-base with origin/develop.
#
# Usage:
#   ./scripts/functional_tests/run_perf_compare.sh [OPTIONS]
#
# Options:
#   --iterations N        Number of test iterations per config (default: 10)
#   --suite SUITE         Test suite: "heatmap", "all", "rma", etc (default: heatmap)
#   --build-config CFG    Build config script name under scripts/build_configs/
#                         (default: all_backends)
#   --cmake-args ARGS     Extra cmake args for all builds (quoted string)
#   --branch-args ARGS    Extra cmake args for the branch build only (quoted string)
#   --variant-args SPEC   Additional named variant: NAME:ENV1=V1,...:cmake-args
#                         NAME      - label used in plots and directory name
#                         ENV1=V1,… - comma-separated env vars set at runtime
#                         cmake-args - extra cmake args for this variant's build
#                         May be repeated for multiple variants.
#                         Example: --variant-args "sdma-on:ROCSHMEM_SDMA_ENABLED=1:-DUSE_SDMA=ON"
#   --pr NUM              Compare GitHub PR NUM against origin/develop baseline.
#                         Fetches the PR branch, builds it, and names it "pr<NUM>".
#   --base-branch NAME    Base branch to compare against (default: origin/develop)
#   --baseline-dir PATH   Use PATH as pre-built baseline (skips baseline build).
#                         Use with --skip-baseline to also skip baseline test runs.
#   --branch-dir PATH     Use PATH as pre-built branch build (skips branch build).
#   --skip-build          Skip all build steps (reuse existing builds)
#   --skip-baseline       Skip baseline build and test runs (reuse existing logs)
#   --skip-develop        Alias for --skip-baseline
#   --skip-branch         Skip branch/PR build and test runs (reuse existing logs)
#   --outdir DIR          Output directory for plots (default: auto-generated)
#
# Prerequisites:
#   - Open MPI in PATH (or set OMPI_HOME)
#   - ROCm installed at /opt/rocm
#   - Python 3.x with ensurepip (python3-venv package on Debian/Ubuntu)
#
# Output:
#   <outdir>/heatmap_summary.png   Summary heatmap
#   <outdir>/per_test/*.png        Per-test latency curves
#   <outdir>/heatmap_data.csv      Raw percentage data
###############################################################################

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
ITERATIONS=10
SUITE=heatmap
BUILD_CONFIG=all_backends
CMAKE_ARGS=""
BRANCH_ARGS=""
VARIANT_SPECS=()   # array of "name:env1=v1,...:cmake-args"
PR_NUM=""
SKIP_BUILD=0
SKIP_BASELINE=0
SKIP_BRANCH=0
BASE_BRANCH="origin/develop"
BASELINE_DIR=""
BRANCH_DIR=""
OUTDIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --iterations)    ITERATIONS="$2";                      shift 2 ;;
    --suite)         SUITE="$2";                           shift 2 ;;
    --build-config)  BUILD_CONFIG="$2";                    shift 2 ;;
    --cmake-args)    CMAKE_ARGS="$2";                      shift 2 ;;
    --branch-args)   BRANCH_ARGS="$2";                     shift 2 ;;
    --variant-args)  VARIANT_SPECS+=("$2");                shift 2 ;;
    --pr)            PR_NUM="$2";                          shift 2 ;;
    --base-branch)   BASE_BRANCH="$2";                     shift 2 ;;
    --baseline-dir)  BASELINE_DIR="$2";                    shift 2 ;;
    --branch-dir)    BRANCH_DIR="$2";                      shift 2 ;;
    --skip-build)    SKIP_BUILD=1;                         shift ;;
    --skip-baseline) SKIP_BASELINE=1;                      shift ;;
    --skip-develop)  SKIP_BASELINE=1;                      shift ;;
    --skip-branch)   SKIP_BRANCH=1;                        shift ;;
    --outdir)        OUTDIR="$2";                          shift 2 ;;
    -h|--help)
      sed -n '2,/^###$/p' "$0" | head -n -1
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROCSHMEM_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECTS_DIR="$(cd "$ROCSHMEM_DIR/.." && pwd)"
DRIVER="$ROCSHMEM_DIR/scripts/functional_tests/driver.sh"
COMPARE="$ROCSHMEM_DIR/scripts/functional_tests/perf_compare.py"
VENV_DIR="$SCRIPT_DIR/.perf-venv"

# Open MPI
export PATH="${OMPI_HOME:-$HOME/ompi}/bin:$PATH"
export LD_LIBRARY_PATH="${OMPI_HOME:-$HOME/ompi}/lib:${LD_LIBRARY_PATH:-}"

# ---------------------------------------------------------------------------
# Determine branch label and build directories
# ---------------------------------------------------------------------------
cd "$ROCSHMEM_DIR"

if [[ -n "$PR_NUM" ]]; then
  # PR mode: compare PR branch against develop baseline
  BRANCH_LABEL="pr${PR_NUM}"
  BRANCH_SAFE="pr${PR_NUM}"
  BUILD_BASELINE="${BASELINE_DIR:-$PROJECTS_DIR/build-develop}"
  BUILD_BRANCH="${BRANCH_DIR:-$PROJECTS_DIR/build-${BRANCH_SAFE}}"
  [[ -z "$OUTDIR" ]] && OUTDIR="$PROJECTS_DIR/plots-${SUITE}-${BRANCH_SAFE}"

  echo "================================================================================"
  echo "Performance Comparison (PR mode)"
  echo "  PR:          #${PR_NUM}"
  echo "  Base branch: $BASE_BRANCH"
  echo "  Suite:       $SUITE"
  echo "  Iterations:  $ITERATIONS"
  echo "  Build cfg:   $BUILD_CONFIG"
  echo "  Baseline:    $BUILD_BASELINE"
  echo "  Branch:      $BUILD_BRANCH"
  echo "  Output:      $OUTDIR"
  echo "================================================================================"
else
  # Default mode: current branch vs merge-base
  BRANCH_NAME="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "detached")"
  if [[ "$BRANCH_NAME" == "HEAD" || "$BRANCH_NAME" == "detached" ]]; then
    BRANCH_NAME="$(git rev-parse --short HEAD)"
  fi

  MERGE_BASE="$(git merge-base HEAD "$BASE_BRANCH" 2>/dev/null)" || {
    echo "ERROR: Cannot find merge-base between HEAD and $BASE_BRANCH" >&2
    echo "       Make sure '$BASE_BRANCH' exists (try: git fetch origin)" >&2
    exit 1
  }
  MERGE_BASE_SHORT="$(git rev-parse --short "$MERGE_BASE")"
  BRANCH_LABEL="$BRANCH_NAME"
  BRANCH_SAFE="$(echo "$BRANCH_NAME" | tr '/' '-' | tr -cd '[:alnum:]-_.')"

  BUILD_BASELINE="${BASELINE_DIR:-$PROJECTS_DIR/build-baseline-$MERGE_BASE_SHORT}"
  BUILD_BRANCH="${BRANCH_DIR:-$PROJECTS_DIR/build-branch-$BRANCH_SAFE}"
  [[ -z "$OUTDIR" ]] && OUTDIR="$PROJECTS_DIR/plots-${SUITE}-${BRANCH_SAFE}"

  echo "================================================================================"
  echo "Performance Comparison"
  echo "  Branch:      $BRANCH_NAME"
  echo "  Base branch: $BASE_BRANCH"
  echo "  Merge-base:  $MERGE_BASE_SHORT"
  echo "  Suite:       $SUITE"
  echo "  Iterations:  $ITERATIONS"
  echo "  Build cfg:   $BUILD_CONFIG"
  echo "  Baseline:    $BUILD_BASELINE"
  echo "  Branch:      $BUILD_BRANCH"
  echo "  Output:      $OUTDIR"
  echo "================================================================================"
fi

# ---------------------------------------------------------------------------
# Step 0: Python virtual environment
# ---------------------------------------------------------------------------
echo ""
echo ">>> Step 0: Setting up Python virtual environment"

# Self-healing: if the venv exists but is missing required packages, recreate it.
if [[ -d "$VENV_DIR" ]] && \
   ! "$VENV_DIR/bin/python3" -c "import matplotlib, numpy, pandas, seaborn" &>/dev/null; then
  echo "  Existing venv is missing dependencies, recreating..."
  rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  # Try a fully isolated venv first (requires python3-venv / ensurepip).
  # Fall back to --system-site-packages --without-pip when ensurepip is
  # unavailable (e.g. Debian/Ubuntu without python3.x-venv installed).
  if python3 -m ensurepip --version &>/dev/null; then
    python3 -m venv "$VENV_DIR" || { echo "ERROR: Failed to create venv." >&2; exit 1; }
  else
    python3 -m venv --system-site-packages --without-pip "$VENV_DIR" || {
      echo "ERROR: Failed to create venv (even without pip)." >&2; exit 1
    }
  fi
  "$VENV_DIR/bin/python3" -m pip install --quiet matplotlib numpy pandas seaborn || {
    echo "ERROR: Failed to install Python dependencies." >&2
    echo "  If using system pip, try: sudo apt install python3-pip" >&2
    exit 1
  }
  echo "  Created venv and installed dependencies at $VENV_DIR"
else
  echo "  Reusing existing venv at $VENV_DIR"
fi

PYTHON="$VENV_DIR/bin/python3"

# ---------------------------------------------------------------------------
# Step 1: Build
# ---------------------------------------------------------------------------

_find_build_config() {
  local worktree="$1"
  local config="$2"
  local result=""
  for candidate in \
    "$worktree/scripts/build_configs/$config" \
    "$worktree/projects/rocshmem/scripts/build_configs/$config"; do
    if [[ -x "$candidate" ]]; then
      result="$candidate"
      break
    fi
  done
  echo "$result"
}

if [[ $SKIP_BUILD -eq 0 ]]; then
  echo ""
  echo ">>> Step 1: Building configurations"

  # Locate branch build config
  BUILD_CONFIG_SCRIPT="$ROCSHMEM_DIR/scripts/build_configs/$BUILD_CONFIG"
  if [[ ! -x "$BUILD_CONFIG_SCRIPT" ]]; then
    echo "ERROR: Build config not found: $BUILD_CONFIG_SCRIPT" >&2
    exit 1
  fi

  # Build baseline
  if [[ $SKIP_BASELINE -eq 0 && -z "$BASELINE_DIR" ]]; then
    echo "--- Building baseline ---"
    WORKTREE="/tmp/rocshmem-baseline-$$"

    if [[ -n "$PR_NUM" ]]; then
      # PR mode: checkout develop
      git -C "$ROCSHMEM_DIR" worktree add "$WORKTREE" "$BASE_BRANCH" --detach 2>&1
    else
      # Default mode: checkout merge-base
      git -C "$ROCSHMEM_DIR" worktree add "$WORKTREE" "$MERGE_BASE" --detach 2>&1
    fi

    BASELINE_CONFIG="$(_find_build_config "$WORKTREE" "$BUILD_CONFIG")"
    if [[ -z "$BASELINE_CONFIG" ]]; then
      echo "ERROR: Cannot find $BUILD_CONFIG in baseline worktree" >&2
      git -C "$ROCSHMEM_DIR" worktree remove "$WORKTREE" 2>/dev/null || true
      exit 1
    fi

    mkdir -p "$BUILD_BASELINE"
    rm -f "$BUILD_BASELINE/CMakeCache.txt"
    (cd "$BUILD_BASELINE" && "$BASELINE_CONFIG" $CMAKE_ARGS)
    git -C "$ROCSHMEM_DIR" worktree remove "$WORKTREE" 2>/dev/null || true
  else
    echo "--- Skipping baseline build ---"
  fi

  # Build branch
  if [[ $SKIP_BRANCH -eq 1 ]]; then
    echo "--- Skipping branch build ---"
  elif [[ -z "$BRANCH_DIR" ]]; then
    if [[ -n "$PR_NUM" ]]; then
      echo "--- Fetching and building PR #${PR_NUM} ---"
      WORKTREE="/tmp/rocshmem-pr${PR_NUM}-$$"
      git -C "$ROCSHMEM_DIR" fetch origin "pull/${PR_NUM}/head"
      git -C "$ROCSHMEM_DIR" worktree add "$WORKTREE" FETCH_HEAD --detach 2>&1
      PR_CONFIG="$(_find_build_config "$WORKTREE" "$BUILD_CONFIG")"
      if [[ -z "$PR_CONFIG" ]]; then
        echo "ERROR: Cannot find $BUILD_CONFIG in PR worktree" >&2
        git -C "$ROCSHMEM_DIR" worktree remove "$WORKTREE" 2>/dev/null || true
        exit 1
      fi
      mkdir -p "$BUILD_BRANCH"
      (cd "$BUILD_BRANCH" && "$PR_CONFIG" $CMAKE_ARGS $BRANCH_ARGS)
      git -C "$ROCSHMEM_DIR" worktree remove "$WORKTREE" 2>/dev/null || true
    else
      echo "--- Building branch ($BRANCH_LABEL) ---"
      mkdir -p "$BUILD_BRANCH"
      (cd "$BUILD_BRANCH" && "$BUILD_CONFIG_SCRIPT" $CMAKE_ARGS $BRANCH_ARGS)
    fi
  else
    echo "--- Using pre-built branch dir: $BRANCH_DIR ---"
  fi

  # Build additional variants
  for spec in "${VARIANT_SPECS[@]}"; do
    IFS=':' read -r vname _venv vcmake <<< "$spec"
    vdir="$PROJECTS_DIR/build-${BRANCH_SAFE}-${vname}"
    echo "--- Building variant: $vname ---"
    mkdir -p "$vdir"
    (cd "$vdir" && "$BUILD_CONFIG_SCRIPT" $CMAKE_ARGS $vcmake)
  done

  echo ">>> Builds complete"
else
  echo ""
  echo ">>> Step 1: Skipping build (--skip-build)"
fi

# ---------------------------------------------------------------------------
# Step 2: Run tests
# ---------------------------------------------------------------------------
echo ""
echo ">>> Step 2: Running $SUITE tests ($ITERATIONS iterations)"

run_iterations() {
  local build_dir="$1"
  local log_prefix="$2"
  local label="$3"
  local env_prefix="${4:-}"

  for i in $(seq 1 "$ITERATIONS"); do
    printf "  %-24s iteration %2d/%d\n" "$label" "$i" "$ITERATIONS"
    (
      cd "$build_dir"
      if [[ -n "$env_prefix" ]]; then
        eval "$env_prefix" "$DRIVER" tests/functional_tests/rocshmem_functional_tests \
          "$SUITE" "${log_prefix}-${i}" 2>&1 | tail -1
      else
        "$DRIVER" tests/functional_tests/rocshmem_functional_tests \
          "$SUITE" "${log_prefix}-${i}" 2>&1 | tail -1
      fi
    )
  done
}

if [[ $SKIP_BASELINE -eq 0 ]]; then
  run_iterations "$BUILD_BASELINE" "logs-${SUITE}" "baseline"
else
  echo "  Skipping baseline test runs (--skip-baseline)"
fi
if [[ $SKIP_BRANCH -eq 0 ]]; then
  run_iterations "$BUILD_BRANCH" "logs-${SUITE}" "${BRANCH_LABEL}"
else
  echo "  Skipping branch test runs (--skip-branch)"
fi

for spec in "${VARIANT_SPECS[@]}"; do
  IFS=':' read -r vname venv _vcmake <<< "$spec"
  vdir="$PROJECTS_DIR/build-${BRANCH_SAFE}-${vname}"
  env_prefix=""
  IFS=',' read -ra pairs <<< "$venv"
  for pair in "${pairs[@]}"; do
    [[ -n "$pair" ]] && env_prefix+="export $pair; "
  done
  run_iterations "$vdir" "logs-${SUITE}" "$vname" "$env_prefix"
done

echo ">>> All test runs complete"

# ---------------------------------------------------------------------------
# Step 3: Generate comparison
# ---------------------------------------------------------------------------
echo ""
echo ">>> Step 3: Generating comparison plots"

# Build --variants args: always include the branch, then extra variants
VARIANT_ARGS=("${BRANCH_LABEL}:$BUILD_BRANCH/logs-${SUITE}-*")
for spec in "${VARIANT_SPECS[@]}"; do
  IFS=':' read -r vname venv vcmake <<< "$spec"
  vdir="$PROJECTS_DIR/build-${BRANCH_SAFE}-${vname}"
  VARIANT_ARGS+=("${vname}:${vdir}/logs-${SUITE}-*")
done

"$PYTHON" "$COMPARE" \
  --baseline "$BUILD_BASELINE/logs-${SUITE}-*" \
  --variants "${VARIANT_ARGS[@]}" \
  --outdir "$OUTDIR"

echo ""
echo "================================================================================"
echo "DONE"
echo "  Heatmap:    $OUTDIR/heatmap_summary.png"
echo "  Summary:    $OUTDIR/heatmap_summary.txt"
echo "  CSV data:   $OUTDIR/heatmap_data.csv"
echo "  Per-test:   $OUTDIR/per_test/  (*.png + *.txt)"
echo "================================================================================"
