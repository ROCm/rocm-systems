#!/bin/bash
###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
#
# Compare wall-clock build time (and per-component ninja time) between two
# commits, the same commit-vs-commit pattern resource_usage_compare.sh /
# lto_inline_remarks_compare.sh use for codegen regressions -- here for "did
# this change make the build slower" instead of "did it change codegen".
#
# Each commit is built in an isolated git worktree under /tmp, so the main
# working tree is never touched -- uncommitted changes are safe, and this
# works against commits that don't even have build_time.sh/this script
# checked in yet: the CURRENT checked-out copy of build_time.sh (not the
# worktree's) wraps whichever build-config script that commit's own worktree
# provides, so only the build step is commit-specific -- the timing/report
# tooling itself always runs at its current, already-fixed-up version.
#
# Builds are cached per (gpu_target, build_config, commit) under
# $PARENT_DIR/build-cache-timed-build and reused by default. Pass
# --force-rebuild for a matched-fresh-pair rebuild of both sides before a
# real before/after decision: wall-clock build time is noisy run-to-run
# (machine load, thermal throttling, filesystem cache state) in a way GPU
# resource usage isn't, so diffing a fresh build against a stale cached one
# (built on a different day, under different load) can show a spurious
# delta that's pure measurement noise. The durable outputs (.ninja_log,
# build_timing.log, build_time-<key>.csv, the full build-time-analysis.md
# report) live separately under $PARENT_DIR/build-time/cache, so
# build-cache-timed-build can be wiped for disk space without losing prior
# measurements.
#
# Usage:
#   ./build_time_compare.sh [OPTIONS]
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
#   --config-path PATH    Direct path to an executable build-config script,
#                         used instead of the --build-config name search.
#                         Relative paths are resolved inside each commit's
#                         isolated worktree (so the version of the script
#                         checked in at that commit is used); absolute
#                         paths are used as-is.
#   --commit1-extra-args ARGS  Extra args appended to the build-config
#                         invocation for commit1 only (e.g. "-DUSE_TDM=1").
#                         Quote it as one string; it is word-split as-is.
#                         Folded into the build/cache key, so this commit's
#                         plain build and its extra-args build are cached
#                         separately.
#   --commit2-extra-args ARGS  Same as --commit1-extra-args, for commit2 only.
#   --time-trace          Add -DROCSHMEM_BUILD_TIME_TRACE=ON to both builds,
#                         so each side's full report includes a -ftime-trace
#                         phase/template-instantiation summary. Adds compile
#                         overhead to both sides equally.
#   --force-rebuild        Force a fresh rebuild+re-measure even if the
#                         commit is already cached (needed after changing
#                         --build-config, --time-trace, or for a
#                         matched-fresh-pair two-commit comparison -- see
#                         header note above).
#   --top N               Number of edges shown in the diff report/chart
#                         (default: 30).
#   --output-dir DIR      Directory to write the comparison report to.
#                         Default:
#                         $PARENT_DIR/build-time/<gpu>-<config>-<sha1>-vs-<sha2>/
#
# Example: compare two explicit commits (reusing cached builds if present)
#   ./build_time_compare.sh --commit1 d48c64f6e --commit2 3caf8d080
#
# Example: compare a PR against its merge-base with develop
#   ./build_time_compare.sh --pr 42 --build-config all_backends
#
# Example: snapshot a single (old) commit, no diff
#   ./build_time_compare.sh --commit1 d48c64f6e
###############################################################################
set -euo pipefail
# Without this, command substitution runs in a subshell with errexit silently
# UNSET, so a failing command inside measure_commit wouldn't stop the script.
shopt -s inherit_errexit

COMMIT_1=""
COMMIT_2=""
GPU_TARGET="gfx950"
BUILD_CONFIG="all_backends"
CONFIG_PATH=""
COMMIT1_EXTRA_ARGS=""
COMMIT2_EXTRA_ARGS=""
PR_NUM=""
BASE_BRANCH="origin/develop"
TIME_TRACE=false
FORCE_REBUILD=false
TOP_N=30
OUTPUT_DIR=""

_need_arg() {
  # $1 = flag name, $2 = remaining arg count (including the flag itself)
  if [[ "$2" -lt 2 ]]; then
    echo "ERROR: $1 requires an argument" >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --commit1)       _need_arg "$1" "$#"; COMMIT_1="$2";      shift 2 ;;
    --commit2)       _need_arg "$1" "$#"; COMMIT_2="$2";      shift 2 ;;
    --pr)            _need_arg "$1" "$#"; PR_NUM="$2";        shift 2 ;;
    --base-branch)   _need_arg "$1" "$#"; BASE_BRANCH="$2";   shift 2 ;;
    --gpu-target)    _need_arg "$1" "$#"; GPU_TARGET="$2";    shift 2 ;;
    --build-config)  _need_arg "$1" "$#"; BUILD_CONFIG="$2";  shift 2 ;;
    --config-path)   _need_arg "$1" "$#"; CONFIG_PATH="$2";   shift 2 ;;
    --commit1-extra-args) _need_arg "$1" "$#"; COMMIT1_EXTRA_ARGS="$2"; shift 2 ;;
    --commit2-extra-args) _need_arg "$1" "$#"; COMMIT2_EXTRA_ARGS="$2"; shift 2 ;;
    --time-trace)    TIME_TRACE=true;    shift ;;
    --force-rebuild) FORCE_REBUILD=true; shift ;;
    --top)           _need_arg "$1" "$#"; TOP_N="$2";         shift 2 ;;
    --output-dir)    _need_arg "$1" "$#"; OUTPUT_DIR="$2";    shift 2 ;;
    -h|--help)
      sed -n '2,/^#####/p' "$0" | head -n -1
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PARENT_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Distinguishes cache/build/output dirs between the two ways of pointing at a
# build, so a --config-path run never collides with (or silently reuses) a
# --build-config run's cache.
if [[ -n "$CONFIG_PATH" ]]; then
  CONFIG_LABEL="path-$(echo "$CONFIG_PATH" | tr '/' '_')"
else
  CONFIG_LABEL="$BUILD_CONFIG"
fi

# This script's own directory -- always the CURRENT checked-out copy of
# build_time.sh/build_time_report.py/build_time_to_csv.py, regardless of
# which commit is being measured. See header note.
TOOLS_DIR="$SCRIPT_DIR"

# Tracks the in-progress worktree so the EXIT trap can remove it if a build
# fails partway through (errexit would otherwise leak the worktree/checkout).
CURRENT_WORKTREE=""
_cleanup_worktree() {
  if [[ -n "$CURRENT_WORKTREE" ]]; then
    git -C "$PROJECT_DIR" worktree remove --force "$CURRENT_WORKTREE" 2>/dev/null || true
    CURRENT_WORKTREE=""
  fi
}
trap _cleanup_worktree EXIT

# _cache_key_suffix ARGS -> short filesystem-safe tag derived from extra build
# args, so "same sha, different -D flags" don't collide in the build/report
# cache dirs. Empty input -> empty output (no change to existing cache keys).
_cache_key_suffix() {
  local args="$1"
  [[ -z "$args" ]] && return
  echo "-$(echo -n "$args" | tr -c 'A-Za-z0-9' '_' | sed 's/_\+/_/g; s/^_//; s/_$//')"
}

# Only used for --build-config (name search); --config-path bypasses this
# entirely. The second candidate is a legacy fallback for rocSHMEM's own
# rocm-systems monorepo nesting -- harmless no-op for any other project.
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

# Resolves the build-config script to invoke for this commit's worktree:
# --config-path (direct path, relative-to-worktree or absolute) takes
# precedence over --build-config (name search under scripts/build_configs/).
_resolve_build_config() {
  local worktree="$1"
  local result=""
  if [[ -n "$CONFIG_PATH" ]]; then
    if [[ "$CONFIG_PATH" = /* ]]; then
      result="$CONFIG_PATH"
    else
      result="$worktree/$CONFIG_PATH"
    fi
    if [[ ! -x "$result" ]]; then
      echo "ERROR: --config-path $CONFIG_PATH is not an executable file ($result)" >&2
      exit 1
    fi
  else
    result="$(_find_build_config "$worktree" "$BUILD_CONFIG")"
    if [[ -z "$result" ]]; then
      echo "ERROR: Cannot find $BUILD_CONFIG in worktree" >&2
      exit 1
    fi
  fi
  echo "$result"
}

# measure_commit <commit> <sha> <extra_build_args> -> prints the path to that
# commit's cached build_time CSV. Also leaves a full build-time-analysis.md
# report (+ timeline/dot-graphs) under the same cache dir when the build
# itself is still available to read .ninja_log/build.ninja/trace JSON from
# (the scratch build dir, not just the durable CSV).
measure_commit() {
  local commit="$1"
  local sha="$2"
  local extra_args="$3"
  local key="${sha}$(_cache_key_suffix "$extra_args")"
  local build_dir="$PARENT_DIR/build-cache-timed-build/${GPU_TARGET}-${CONFIG_LABEL}-${key}"
  local cache_dir="$PARENT_DIR/build-time/cache/${GPU_TARGET}-${CONFIG_LABEL}-${key}"
  local csv="$cache_dir/build_time-${key}.csv"
  local report_md="$cache_dir/report/build-time-analysis.md"

  if [[ -f "$csv" && "$FORCE_REBUILD" == false ]]; then
    echo "  [$key] cached -> $csv" >&2
  else
    echo "  [$key] building ($GPU_TARGET / $CONFIG_LABEL${extra_args:+ / $extra_args})..." >&2
    local worktree="/tmp/timed-build-worktree-${key}-$$"

    git -C "$PROJECT_DIR" worktree add "$worktree" "$commit" --detach >&2
    CURRENT_WORKTREE="$worktree"

    local FOUND_BUILD_CONFIG
    FOUND_BUILD_CONFIG="$(_resolve_build_config "$worktree")"

    # cmake --fresh is unreliable at fully resetting state between commits, so
    # wipe the build dir ourselves.
    rm -rf "$build_dir"
    mkdir -p "$build_dir"
    mkdir -p "$cache_dir"

    local time_trace_flag=""
    [[ "$TIME_TRACE" == true ]] && time_trace_flag="-DROCSHMEM_BUILD_TIME_TRACE=ON"

    (
      cd "$build_dir"
      # $extra_args/$time_trace_flag are intentionally unquoted/word-split
      # here so the caller can pass e.g. "-DUSE_TDM=1 -DUSE_SDMA=1" as one
      # string and have it expand into separate cmake args.
      # shellcheck disable=SC2086
      "$TOOLS_DIR/build_time.sh" "$FOUND_BUILD_CONFIG" \
        -DGPU_TARGETS="$GPU_TARGET" $time_trace_flag $extra_args 2>&1 |
        tee "$cache_dir/build.log" >&2
    )

    git -C "$PROJECT_DIR" worktree remove "$worktree" >&2 || true
    CURRENT_WORKTREE=""

    python3 "$TOOLS_DIR/build_time_to_csv.py" \
      --build-dir "$build_dir" --gpu-target "$GPU_TARGET" --build-config "$CONFIG_LABEL" \
      --commit "$key" --out "$csv" --top 0 >&2
  fi

  if [[ ! -f "$report_md" || "$FORCE_REBUILD" == true ]]; then
    if [[ -f "$build_dir/.ninja_log" ]]; then
      mkdir -p "$cache_dir/report"
      python3 "$TOOLS_DIR/build_time_report.py" \
        --build-dir "$build_dir" --output-dir "$cache_dir/report/diagrams" \
        --report "$report_md" --top "$TOP_N" >&2
    else
      echo "  [$key] note: $build_dir/.ninja_log missing -- skipping full" >&2
      echo "  build-time-analysis.md regen (build_time-${key}.csv is still valid)." >&2
      echo "  Rerun with --force-rebuild to get a full report." >&2
    fi
  fi

  echo "$csv"
}

if [[ -n "$PR_NUM" ]]; then
  echo "  Fetching PR #${PR_NUM}..." >&2
  git -C "$PROJECT_DIR" fetch origin "pull/${PR_NUM}/head"
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

# Cached builds are reused by default; nudge towards --force-rebuild for a
# real before/after decision, since wall-clock build time is noisy
# run-to-run and a stale baseline can show a spurious delta.
if [[ -n "$COMMIT_2" && "$FORCE_REBUILD" == false ]]; then
  echo "  Two-commit comparison: reusing cached builds where available." >&2
  echo "  Pass --force-rebuild for a matched-fresh-pair rebuild before a real" >&2
  echo "  before/after decision (wall-clock build time is noisy run-to-run)." >&2
fi

echo "=== build time: $COMMIT_1${COMMIT_2:+ vs $COMMIT_2} ($GPU_TARGET / $CONFIG_LABEL) ==="

SHA_1="$(git rev-parse --short=12 "$COMMIT_1")"
CSV_1="$(measure_commit "$COMMIT_1" "$SHA_1" "$COMMIT1_EXTRA_ARGS")"
KEY_1="${SHA_1}$(_cache_key_suffix "$COMMIT1_EXTRA_ARGS")"
CACHE_DIR_1="$PARENT_DIR/build-time/cache/${GPU_TARGET}-${CONFIG_LABEL}-${KEY_1}"

if [[ -z "$COMMIT_2" ]]; then
  echo ""
  echo "Single-commit snapshot -> $CSV_1"
  if [[ -f "$CACHE_DIR_1/report/build-time-analysis.md" ]]; then
    echo "  Full report -> $CACHE_DIR_1/report/build-time-analysis.md"
    echo "  Timeline/dot graphs -> $CACHE_DIR_1/report/diagrams/"
  fi
  exit 0
fi

SHA_2="$(git rev-parse --short=12 "$COMMIT_2")"
CSV_2="$(measure_commit "$COMMIT_2" "$SHA_2" "$COMMIT2_EXTRA_ARGS")"
KEY_2="${SHA_2}$(_cache_key_suffix "$COMMIT2_EXTRA_ARGS")"
CACHE_DIR_2="$PARENT_DIR/build-time/cache/${GPU_TARGET}-${CONFIG_LABEL}-${KEY_2}"

OUTDIR="${OUTPUT_DIR:-$PARENT_DIR/build-time/${GPU_TARGET}-${CONFIG_LABEL}-${KEY_1}-vs-${KEY_2}}"
mkdir -p "$OUTDIR"
cp "$CSV_1" "$OUTDIR/build_time-${KEY_1}.csv"
cp "$CSV_2" "$OUTDIR/build_time-${KEY_2}.csv"

python3 "$TOOLS_DIR/build_time_diff.py" \
  --baseline "$CSV_1" --branch "$CSV_2" \
  --out "$OUTDIR/build_time_diff.csv" --chart "$OUTDIR/build_time_diff.png" \
  --top "$TOP_N" | tee "$OUTDIR/build_time_diff.txt"

echo ""
echo "Done. Report -> $OUTDIR/"
echo "  Per-commit CSVs: build_time-${KEY_1}.csv, build_time-${KEY_2}.csv"
echo "  Diff:            build_time_diff.{csv,png,txt}"
if [[ -f "$CACHE_DIR_1/report/build-time-analysis.md" && -f "$CACHE_DIR_2/report/build-time-analysis.md" ]]; then
  echo "  Full per-commit reports (left where they were built, not copied, so"
  echo "  their timeline.json/dot-graph diagrams stay alongside them):"
  echo "    $CACHE_DIR_1/report/build-time-analysis.md"
  echo "    $CACHE_DIR_2/report/build-time-analysis.md"
fi
