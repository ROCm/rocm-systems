#!/bin/bash
###############################################################################
# Compare per-kernel GPU resource usage (VGPR/SGPR/AGPR/scratch/LDS/occupancy)
# between two commits (or one commit vs the current working tree).
#
# Builds are cached per (gpu_target, build_config, commit) under
# $PROJECTS_DIR/build-resource-usage-* so re-comparing COMMIT_1 against a
# different COMMIT_2 doesn't rebuild COMMIT_1 again.
#
# Each commit is built in an isolated git worktree under /tmp so the main
# working tree is never touched — uncommitted changes are safe.
#
# Usage:
#   ./resource_usage_compare.sh [OPTIONS]
#
# Options:
#   --commit1 REF         First commit/branch to measure (default: HEAD, or
#                         merge-base with --base-branch when --pr is set).
#   --commit2 REF         Second commit/branch to compare against commit1.
#                         Omit to just snapshot commit1 with no diff.
#   --pr NUM              Fetch GitHub PR #NUM and compare it against its
#                         merge-base with --base-branch. Sets commit2=FETCH_HEAD
#                         and commit1=merge-base unless overridden.
#   --base-branch NAME    Base branch for merge-base resolution with --pr
#                         (default: origin/develop).
#   --gpu-target ARCH     GPU target architecture (default: gfx950).
#   --build-config CFG    Build config script under scripts/build_configs/
#                         (default: all_backends).
#   --skip-build          Reuse whatever's already cached, just regenerate the
#                         diff report/chart (fast iteration on report format).
#   --force-rebuild       Rebuild+re-extract even if the commit is already
#                         cached (needed after changing --build-config or the
#                         resource-usage extraction scripts themselves).
#   --match REGEX         Pin kernels matching this regex (against demangled or
#                         mangled name, case-insensitive) to the top of every
#                         report/chart regardless of delta.
#
# Example: compare two explicit commits
#   ./resource_usage_compare.sh --commit1 d48c64f6e --commit2 3caf8d080 \
#     --build-config all_backends --force-rebuild
#
# Example: compare a PR against its merge-base with develop
#   ./resource_usage_compare.sh --pr 42 --build-config all_backends
#
# Example: pin a specific kernel to the top of every report/chart
#   ./resource_usage_compare.sh --commit1 673440d --commit2 da18d28 \
#     --match alltoall_test
###############################################################################
set -euo pipefail
# Without this, command substitution $(...) (e.g. CSV_1="$(measure_commit ...)")
# runs in a subshell with errexit silently UNSET, so a failing command inside
# measure_commit (e.g. python3 erroring out) does not stop the script -- it
# just falls through to `echo "$csv"`, which exits 0 and masks the failure.
shopt -s inherit_errexit

COMMIT_1=""
COMMIT_2=""
GPU_TARGET="gfx950"
BUILD_CONFIG="all_backends"
PR_NUM=""
BASE_BRANCH="origin/develop"
SKIP_BUILD=false
FORCE_REBUILD=false
MATCH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --commit1)       COMMIT_1="$2";      shift 2 ;;
    --commit2)       COMMIT_2="$2";      shift 2 ;;
    --pr)            PR_NUM="$2";        shift 2 ;;
    --base-branch)   BASE_BRANCH="$2";   shift 2 ;;
    --gpu-target)    GPU_TARGET="$2";    shift 2 ;;
    --build-config)  BUILD_CONFIG="$2";  shift 2 ;;
    --skip-build)    SKIP_BUILD=true;    shift ;;
    --force-rebuild) FORCE_REBUILD=true; shift ;;
    --match)         MATCH="$2";         shift 2 ;;
    -h|--help)
      sed -n '2,/^#####/p' "$0" | head -n -1
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
ROCSHMEM_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECTS_DIR="$(cd "$ROCSHMEM_DIR/.." && pwd)"
cd "$ROCSHMEM_DIR"


TOOLS_DIR="$ROCSHMEM_DIR/scripts/functional_tests"

# Tracks the worktree currently being built so the EXIT trap below can clean
# it up if the build fails partway through (errexit exits the script before
# reaching the normal `git worktree remove` call, otherwise leaking a
# worktree registration + /tmp checkout).
CURRENT_WORKTREE=""
_cleanup_worktree() {
  if [[ -n "$CURRENT_WORKTREE" ]]; then
    git -C "$ROCSHMEM_DIR" worktree remove --force "$CURRENT_WORKTREE" 2>/dev/null || true
    CURRENT_WORKTREE=""
  fi
}
trap _cleanup_worktree EXIT

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

# measure_commit <commit> -> prints the path to that commit's cached CSV
measure_commit() {
  local commit="$1"
  local sha="$2"
  local build_dir="$PROJECTS_DIR/build-resource-usage-${GPU_TARGET}-${BUILD_CONFIG}-${sha}"
  local csv="$build_dir/res-${sha}.csv"

  if [[ -f "$csv" && "$FORCE_REBUILD" == false ]]; then
    echo "  [$sha] cached -> $csv" >&2
    echo "$csv"
    return
  fi

  if [[ "$SKIP_BUILD" == true ]]; then
    echo "ERROR: --skip-build set but no cached CSV for $sha at $csv" >&2
    exit 1
  fi

  echo "  [$sha] building ($GPU_TARGET / $BUILD_CONFIG)..." >&2
  local worktree="/tmp/rocshmem-resource-usage-${sha}-$$"

  git -C "$ROCSHMEM_DIR" worktree add "$worktree" "$commit" --detach >&2
  CURRENT_WORKTREE="$worktree"

  FOUND_BUILD_CONFIG="$(_find_build_config "$worktree" "$BUILD_CONFIG")"
  if [[ -z "$FOUND_BUILD_CONFIG" ]]; then
    echo "ERROR: Cannot find $BUILD_CONFIG in baseline worktree" >&2
    exit 1
  fi

  # cmake's --fresh has been unreliable at fully resetting cache/generated
  # state between commits, so wipe the directory ourselves instead of
  # relying on it.
  rm -rf "$build_dir"
  mkdir -p "$build_dir"
  (
    cd "$build_dir"
    # measure_commit's own stdout is captured by the caller ($(measure_commit ...)) and
    # must contain only the final `echo "$csv"` path below -- tee's stdout copy of the
    # build log must go to stderr (>&2), not stdout, or it corrupts the captured path
    # (and can make it megabytes long, blowing out ARG_MAX in later `cp "$CSV_1" ...`).
    "$FOUND_BUILD_CONFIG" \
      --fresh \
      -DGPU_TARGETS="$GPU_TARGET" \
      -DCMAKE_CXX_FLAGS="-Rpass-analysis=kernel-resource-usage" 2>&1 |
      tee "$build_dir/build.log" >&2
    grep -B1 -A9 "Function Name:" "$build_dir/build.log" >"$build_dir/resource_usage_summary.log" || true
  )

  git -C "$ROCSHMEM_DIR" worktree remove "$worktree" >&2 || true
  CURRENT_WORKTREE=""

  # measure_commit's stdout is captured by the caller (CSV_1="$(measure_commit ...)")
  # and must contain only the final `echo "$csv"` path -- redirect this script's own
  # report (which prints to stdout) to stderr so it stays visible without corrupting
  # the captured path.
  python3 "$TOOLS_DIR/resource_usage_to_csv.py" \
    --log "$build_dir/resource_usage_summary.log" \
    --arch "$GPU_TARGET" --build-config "$BUILD_CONFIG" --commit "$sha" \
    --out "$csv" >&2

  echo "$csv"
}

if [[ -n "$PR_NUM" ]]; then
  echo "  Fetching PR #${PR_NUM}..." >&2
  git -C "$ROCSHMEM_DIR" fetch origin "pull/${PR_NUM}/head"
  COMMIT_2="${COMMIT_2:-FETCH_HEAD}"
  if [[ -z "$COMMIT_1" ]]; then
    COMMIT_1="$(git merge-base FETCH_HEAD "$BASE_BRANCH")" || {
      echo "ERROR: Cannot find merge-base between PR #${PR_NUM} and $BASE_BRANCH" >&2
      echo "       Make sure '$BASE_BRANCH' exists (try: git fetch origin)" >&2
      exit 1
    }
  fi
else
  COMMIT_1="${COMMIT_1:-HEAD}"
fi

echo "=== resource usage: $COMMIT_1${COMMIT_2:+ vs $COMMIT_2} ($GPU_TARGET / $BUILD_CONFIG) ==="

SHA_1="$(git rev-parse --short=12 "$COMMIT_1")"
CSV_1="$(measure_commit "$COMMIT_1" "$SHA_1")"

if [[ -z "$COMMIT_2" ]]; then
  echo ""
  echo "Single-commit snapshot -> $CSV_1"
  exit 0
fi

SHA_2="$(git rev-parse --short=12 "$COMMIT_2")"
CSV_2="$(measure_commit "$COMMIT_2" "$SHA_2")"

OUTDIR="$ROCSHMEM_DIR/resource-usage-${GPU_TARGET}-${BUILD_CONFIG}-${SHA_1}-vs-${SHA_2}"
mkdir -p "$OUTDIR"
cp "$CSV_1" "$OUTDIR/res-${SHA_1}.csv"
cp "$CSV_2" "$OUTDIR/res-${SHA_2}.csv"

SORT_BY_TYPE=(
  "VGPRs" "TotalSGPRs" "AGPRs" "ScratchBytesPerLane"
  "OccupancyWavesPerSIMD" "SGPRsSpill" "VGPRsSpill" "LDSBytesPerBlock"
)

for sort_by in "${SORT_BY_TYPE[@]}"; do
  python3 "$TOOLS_DIR/resource_usage_diff.py" \
    --baseline "$CSV_1" \
    --branch "$CSV_2" \
    --out "$OUTDIR/res_diff_${sort_by}.csv" \
    --chart "$OUTDIR/res_diff_${sort_by}.png" \
    --top 20 --sort-by "$sort_by" \
    ${MATCH:+--match "$MATCH"}
done

echo ""
echo "Done. Self-contained report -> $OUTDIR/"
echo "  inputs:  res-${SHA_1}.csv, res-${SHA_2}.csv"
echo "  reports: res_diff_<Column>.{csv,png} for each of: ${SORT_BY_TYPE[*]}"
