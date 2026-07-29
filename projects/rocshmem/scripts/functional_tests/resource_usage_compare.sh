#!/bin/bash
###############################################################################
# Compare per-kernel GPU resource usage (VGPR/SGPR/AGPR/scratch/LDS/occupancy)
# between two commits (or one commit vs the current working tree).
#
# This is the git-checkout-based counterpart to measuring the working tree in
# place with no checkout. Use this script when you need to bisect a
# regression across history or compare a PR branch against its merge-base,
# rather than iterating on local edits.
#
# Builds are cached per (gpu_target, build_config, commit) under
# resource-usage-cache/ so re-comparing COMMIT_1 against a different COMMIT_2
# doesn't rebuild COMMIT_1 again.
#
# Usage:
#   Edit COMMIT_1 / COMMIT_2 / GPU_TARGET / BUILD_CONFIG below, then:
#     ./resource_usage_compare.sh
#   or override via env without editing the file:
#     COMMIT_1=develop COMMIT_2=HEAD ./resource_usage_compare.sh
#
# Env knobs:
#   SKIP_BUILD=true     reuse whatever's already cached, just regenerate the
#                       diff report/chart (fast iteration on report format).
#   FORCE_REBUILD=true  rebuild+re-extract even if the commit is already
#                       cached (needed after changing BUILD_CONFIG or the
#                       resource-usage extraction scripts themselves).
#   MATCH=<regex>       pin kernels matching this regex (against demangled or
#                       mangled name, case-insensitive) to the top of every
#                       report/chart regardless of delta -- use this to keep
#                       an eye on a specific kernel instead of whatever has
#                       the largest delta, e.g. MATCH='alltoall_test'.
#
# Leave COMMIT_2 empty ("") to just measure COMMIT_1 with no comparison.
#
# IMPORTANT: this script does `git checkout --force --detach` on the repo you
# run it from. Stash or commit your working-tree changes first — it will
# silently check out over uncommitted edits.
#
# Example:
# COMMIT_1=d48c64f6e COMMIT_2=3caf8d080 BUILD_CONFIG=all_backends \
#  FORCE_REBUILD=true ./resource_usage_compare.sh
#
# Example, pinning a specific kernel to the top of every report/chart:
# COMMIT_1=673440d COMMIT_2=da18d28 BUILD_CONFIG=all_backends \
#  MATCH='alltoall_test' ./resource_usage_compare.sh
###############################################################################
set -euo pipefail

COMMIT_1="${COMMIT_1:-HEAD}"
COMMIT_2="${COMMIT_2:-}"
GPU_TARGET="${GPU_TARGET:-gfx950}"
BUILD_CONFIG="${BUILD_CONFIG:-all_backends}"

SKIP_BUILD="${SKIP_BUILD:-false}"
FORCE_REBUILD="${FORCE_REBUILD:-false}"
MATCH="${MATCH:-}"

SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
ROCSHMEM_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROCSHMEM_DIR"

if [[ "$SKIP_BUILD" != "true" ]] && ! git diff --quiet HEAD --; then
  echo "ERROR: tracked files have uncommitted changes. This script runs 'git checkout'" >&2
  echo "and will silently discard them. Stash/commit first, or set SKIP_BUILD=true" >&2
  echo "if you only want to regenerate the diff report from cached builds." >&2
  echo "(untracked files are not affected by checkout and are ignored by this check)" >&2
  exit 1
fi

ORIGINAL_REF="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$ORIGINAL_REF" == "HEAD" ]]; then
  ORIGINAL_REF="$(git rev-parse HEAD)"
fi

CACHE_ROOT="$ROCSHMEM_DIR/resource-usage-cache"
mkdir -p "$CACHE_ROOT"

# measure_commit <commit> -> prints the path to that commit's cached CSV
measure_commit() {
  local commit="$1"
  local sha
  sha="$(git rev-parse --short=12 "$commit")"
  local cache_dir="$CACHE_ROOT/${GPU_TARGET}-${BUILD_CONFIG}-${sha}"
  local csv="$cache_dir/res-${sha}.csv"

  if [[ -f "$csv" && "$FORCE_REBUILD" != "true" ]]; then
    echo "  [$sha] cached -> $csv" >&2
    echo "$csv"
    return
  fi

  if [[ "$SKIP_BUILD" == "true" ]]; then
    echo "ERROR: SKIP_BUILD=true but no cached CSV for $sha at $csv" >&2
    exit 1
  fi

  echo "  [$sha] building ($GPU_TARGET / $BUILD_CONFIG)..." >&2
  mkdir -p "$cache_dir"
  local build_dir="$cache_dir/build"

  git checkout --quiet --force --detach "$(git rev-parse "$commit")"

  mkdir -p "$build_dir"
  rm -rf "${build_dir:?}"/*
  (
    cd "$build_dir"
    # measure_commit's own stdout is captured by the caller ($(measure_commit ...)) and
    # must contain only the final `echo "$csv"` path below -- tee's stdout copy of the
    # build log must go to stderr (>&2), not stdout, or it corrupts the captured path
    # (and can make it megabytes long, blowing out ARG_MAX in later `cp "$CSV_1" ...`).
    "$ROCSHMEM_DIR/scripts/build_configs/$BUILD_CONFIG" \
      -DGPU_TARGETS="$GPU_TARGET" \
      -DCMAKE_CXX_FLAGS="-Rpass-analysis=kernel-resource-usage" 2>&1 |
      tee "$cache_dir/build.log" >&2
    grep -B1 -A9 "Function Name:" "$cache_dir/build.log" >"$cache_dir/resource_usage_summary.log" || true
  )

  # measure_commit's stdout is captured by the caller (CSV_1="$(measure_commit ...)")
  # and must contain only the final `echo "$csv"` path -- redirect this script's own
  # report (which prints to stdout) to stderr so it stays visible without corrupting
  # the captured path.
  python3 "$ROCSHMEM_DIR/scripts/functional_tests/resource_usage_to_csv.py" \
    --log "$cache_dir/resource_usage_summary.log" \
    --arch "$GPU_TARGET" --build-config "$BUILD_CONFIG" --commit "$sha" \
    --out "$csv" >&2

  # Build artifacts are large and reproducible from the log; drop them but
  # keep the log + CSV (cheap, and enough to regenerate the CSV if its format
  # changes) so re-running with FORCE_REBUILD=false stays fast.
  rm -rf "$build_dir"

  echo "$csv"
}

echo "=== resource usage: $COMMIT_1${COMMIT_2:+ vs $COMMIT_2} ($GPU_TARGET / $BUILD_CONFIG) ==="

CSV_1="$(measure_commit "$COMMIT_1")"
SHA_1="$(git rev-parse --short=12 "$COMMIT_1")"

git checkout --quiet --force "$ORIGINAL_REF" 2>/dev/null || true

if [[ -z "$COMMIT_2" ]]; then
  echo ""
  echo "Single-commit snapshot -> $CSV_1"
  exit 0
fi

CSV_2="$(measure_commit "$COMMIT_2")"
SHA_2="$(git rev-parse --short=12 "$COMMIT_2")"

git checkout --quiet --force "$ORIGINAL_REF" 2>/dev/null || true

OUTDIR="$ROCSHMEM_DIR/resource-usage-${GPU_TARGET}-${BUILD_CONFIG}-${SHA_1}-vs-${SHA_2}"
mkdir -p "$OUTDIR"
cp "$CSV_1" "$OUTDIR/res-${SHA_1}.csv"
cp "$CSV_2" "$OUTDIR/res-${SHA_2}.csv"

SORT_BY_TYPE=(
  "VGPRs" "TotalSGPRs" "AGPRs" "ScratchBytesPerLane"
  "OccupancyWavesPerSIMD" "SGPRsSpill" "VGPRsSpill" "LDSBytesPerBlock"
)
for sort_by in "${SORT_BY_TYPE[@]}"; do
  python3 "$ROCSHMEM_DIR/scripts/functional_tests/resource_usage_diff.py" \
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
