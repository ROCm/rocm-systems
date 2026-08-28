#!/bin/bash
###############################################################################
# Compare per-kernel GPU resource usage (VGPR/SGPR/AGPR/scratch/LDS/occupancy)
# between two commits (or one commit vs the current working tree).
#
# Builds are cached per (gpu_target, build_config, commit) under
# $PARENT_DIR/build-cache and reused by default. Pass --force-rebuild for a
# "matched-fresh-pair" rebuild of both sides before a real before/after
# decision: AMDGPU LTO codegen (register allocation, scheduling, symbol
# layout) is NON-DETERMINISTIC build-to-build even for identical source, so
# diffing a fresh build against a stale cached one can show spurious deltas
# that are pure build noise. The durable outputs of a build (res-<sha>.csv,
# build.log, resource_usage_summary.log) live separately under
# $PARENT_DIR/resource-usage/cache, so build-cache can be wiped for disk
# space without losing prior measurements.
#
# Each commit is built in an isolated git worktree under /tmp so the main
# working tree is never touched — uncommitted changes are safe.
#
# `rocshmem_device_bitcode` (DeviceBitcode.cmake's librocshmem_device_<arch>.bc,
# consumed by rocshmem_hipmodule_init/Triton-PyTorch JIT) is declared ALL, so
# every build above already produces it as a side effect. Its final `opt -O3`
# is an ungated whole-program inliner, a materially different regime from the
# production library's cost-gated LTO -- this script also backend-compiles
# that .bc with -Rpass-analysis=kernel-resource-usage (see
# measure_device_bitcode() below) and diffs it separately
# (res_diff_bitcode_<Column>.{csv,png}), so a change that looks safe under the
# production library alone doesn't get treated as validated everywhere.
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
#   --config-path PATH    Direct path to an executable build-config script,
#                         used instead of the --build-config name search.
#                         Relative paths are resolved inside each commit's
#                         isolated worktree (so the version of the script
#                         checked in at that commit is used); absolute
#                         paths are used as-is. The script is invoked the
#                         same way a --build-config script is (same cwd,
#                         same appended -D... args) -- see "Portability"
#                         below.
#   --skip-build          Reuse cached builds when available (default
#                         behavior; flag accepted as a no-op for
#                         explicitness).
#   --force-rebuild       Force a fresh rebuild+re-extract even if the
#                         commit is already cached (needed after changing
#                         --build-config or the resource-usage extraction
#                         scripts themselves, or for a matched-fresh-pair
#                         two-commit comparison -- see header note above).
#   --match REGEX         Pin kernels matching this regex (against demangled or
#                         mangled name, case-insensitive) to the top of every
#                         report/chart regardless of delta.
#   --top N               Number of rows to show in each generated chart/CSV
#                         (default: 50).
#   --output-dir DIR      Directory to write the comparison report (CSVs +
#                         charts) to. Default:
#                         $PARENT_DIR/resource-usage/<gpu>-<config>-<sha1>-vs-<sha2>/
#
# Also generates one self-contained interactive HTML dashboard per variant
# (resource_dashboard.html, and resource_dashboard_bitcode.html when bitcode
# data is available; always on -- no flag needed): a baseline-vs-branch comparison table on top
# (scrollable, filterable, switchable by metric) for two-commit runs, plus a
# per-commit stats/chart/table view with an in-page commit selector.
#
# Example: compare two explicit commits (reusing cached builds if present)
#   ./resource_usage_compare.sh --commit1 d48c64f6e --commit2 3caf8d080 \
#     --build-config all_backends
#
# Example: compare a PR against its merge-base with develop
#   ./resource_usage_compare.sh --pr 42 --build-config all_backends
#
# Example: pin a specific kernel to the top of every report/chart
#   ./resource_usage_compare.sh --commit1 673440d --commit2 da18d28 \
#     --match alltoall_test
#
# Portability: this whole scripts/analysis/ directory can be copied into any
# other CMake project, dropped in next to that project's top-level
# CMakeLists.txt (i.e. <project>/scripts/analysis/*), and pointed at that
# project's own build via --config-path (a build script anywhere in the
# repo, no directory convention required) instead of --build-config (which
# stays rocSHMEM's own scripts/build_configs/ name search, used by default).
# The device-bitcode measurement below (measure_device_bitcode) is a
# rocSHMEM-specific extra check for librocshmem_device_<arch>.bc and
# silently no-ops for any project that doesn't produce a matching artifact.
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
PR_NUM=""
BASE_BRANCH="origin/develop"
SKIP_BUILD=true
FORCE_REBUILD=false
MATCH=""
TOP_N=50
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
    --skip-build)    SKIP_BUILD=true;    shift ;;
    --force-rebuild) FORCE_REBUILD=true; shift ;;
    --match)         _need_arg "$1" "$#"; MATCH="$2";         shift 2 ;;
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

# Distinguishes cache/build/output dirs and the CSV build-config tag between
# the two ways of pointing at a build, so a --config-path run never collides
# with (or silently reuses) a --build-config run's cache -- see "Portability"
# note above.
if [[ -n "$CONFIG_PATH" ]]; then
  CONFIG_LABEL="path-$(echo "$CONFIG_PATH" | tr '/' '_')"
else
  CONFIG_LABEL="$BUILD_CONFIG"
fi

TOOLS_DIR="$PROJECT_DIR/scripts/analysis"

# Must match resource_usage_diff.py's NUMERIC_COLS -- these are the valid
# --sort-by values. Also duplicated in .claude/skills/rocshmem-resource-usage/
# scripts/compare.sh; keep both lists in sync.
SORT_BY_TYPE=(
  "VGPRs" "TotalSGPRs" "AGPRs" "ScratchBytesPerLane"
  "OccupancyWavesPerSIMD" "SGPRsSpill" "VGPRsSpill" "LDSBytesPerBlock"
)

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

# Backend-compile the device-bitcode artifact (DeviceBitcode.cmake's
# librocshmem_device_<arch>.bc, already built as a side effect of ALL) with
# -Rpass-analysis=kernel-resource-usage to get its resource-usage remarks.
# DeviceBitcode.cmake's own pipeline never runs AMDGPU instruction selection
# (that normally only happens later, at JIT time), so this reuses those same
# codegen decisions early, for reporting.
#
# The .bc has no debug info, so source_file/line in the output CSV are the
# placeholder "<bitcode>"/0 for every row -- only mangled_name (and the
# resource columns) are meaningful; resource_usage_diff.py keys/compares by
# (arch, build_config, mangled_name), so that's sufficient.
measure_device_bitcode() {
  local bc_file="$1" arch="$2" build_config="$3" commit="$4" out_csv="$5"

  # Same search order as DeviceBitcode.cmake's find_program(LLVM_CLANG ...)
  # plus ROCM_HOME, so this uses the same ROCm clang++ that built the .bc.
  local -a _clangxx_candidates=()
  [[ -n "${ROCM_PATH:-}" ]] && _clangxx_candidates+=("$ROCM_PATH/llvm/bin/clang++")
  [[ -n "${ROCM_HOME:-}" ]] && _clangxx_candidates+=("$ROCM_HOME/llvm/bin/clang++")
  [[ -n "${THEROCK_TOOLCHAIN_ROOT:-}" ]] && \
    _clangxx_candidates+=("$THEROCK_TOOLCHAIN_ROOT/lib/llvm/bin/clang++")

  local clangxx="" _candidate
  for _candidate in "${_clangxx_candidates[@]}"; do
    if [[ -x "$_candidate" ]]; then
      clangxx="$_candidate"
      break
    fi
  done
  if [[ -z "$clangxx" ]]; then
    clangxx="$(command -v clang++)" || {
      echo "error: clang++ not found! (need a ROCm clang++ with AMDGPU backend support)" >&2
      return 1
    }
  fi

  local workdir
  workdir="$(mktemp -d /tmp/device-bitcode-XXXXXX)"
  # RETURN traps aren't scoped to this function -- clear it as it fires, or
  # it re-fires on the caller's next return once $workdir is out of scope.
  trap 'rm -rf "$workdir"; trap - RETURN' RETURN

  local raw_log="$workdir/raw.log"
  local summary_log="$workdir/resource_usage_summary.log"

  echo "  [device-bitcode] backend-compiling $bc_file (mcpu=$arch) for resource-usage remarks..." >&2
  "$clangxx" -target amdgcn-amd-amdhsa -mcpu="$arch" \
    -Rpass-analysis=kernel-resource-usage \
    -c "$bc_file" -o "$workdir/out.o" 2>"$raw_log" || {
      echo "error: backend compile of $bc_file failed:" >&2
      cat "$raw_log" >&2
      return 1
    }

  # Normalize the backend-only remark format into the `<file>:<line>:<col>:
  # Key: value` shape resource_usage_to_csv.py expects.
  sed -E \
    -e 's/^remark: <unknown>:0:0:/<bitcode>:0:0:/' \
    -e 's/ \[-Rpass-analysis=kernel-resource-usage\]$//' \
    "$raw_log" | grep -E '<bitcode>:0:0:' > "$summary_log" || true

  if [[ ! -s "$summary_log" ]]; then
    echo "error: no kernel-resource-usage remarks found compiling $bc_file" >&2
    echo "--- raw compiler output ---" >&2
    cat "$raw_log" >&2
    return 1
  fi

  python3 "$TOOLS_DIR/resource_usage_to_csv.py" \
    --log "$summary_log" \
    --arch "$arch" --build-config "${build_config}-bitcode" --commit "$commit" \
    --out "$out_csv" --top 0 >&2
}

# measure_commit <commit> -> prints the path to that commit's cached CSV
measure_commit() {
  local commit="$1"
  local sha="$2"
  local build_dir="$PARENT_DIR/build-cache/${GPU_TARGET}-${CONFIG_LABEL}-${sha}"
  local cache_dir="$PARENT_DIR/resource-usage/cache/${GPU_TARGET}-${CONFIG_LABEL}-${sha}"
  local csv="$cache_dir/res-${sha}.csv"

  if [[ -f "$csv" && "$FORCE_REBUILD" == false ]]; then
    echo "  [$sha] cached -> $csv" >&2
    echo "$csv"
    return
  fi

  echo "  [$sha] building ($GPU_TARGET / $CONFIG_LABEL)..." >&2
  local worktree="/tmp/resource-usage-worktree-${sha}-$$"

  git -C "$PROJECT_DIR" worktree add "$worktree" "$commit" --detach >&2
  CURRENT_WORKTREE="$worktree"

  local FOUND_BUILD_CONFIG
  FOUND_BUILD_CONFIG="$(_resolve_build_config "$worktree")"

  # cmake --fresh is unreliable at fully resetting state between commits, so
  # wipe the build dir ourselves.
  rm -rf "$build_dir"
  mkdir -p "$build_dir"
  mkdir -p "$cache_dir"
  # resource_usage_to_csv.py merges into any pre-existing --out file; without
  # this, a stale CSV from a prior run of this same commit would silently mix
  # with this fresh build's rows instead of being replaced.
  rm -f "$csv" "$cache_dir/res-${sha}-bitcode.csv"
  (
    cd "$build_dir"
    # Caller captures this function's stdout as the CSV path -- the build
    # log must go to stderr only, or it corrupts (and bloats) that capture.
    # build.log/resource_usage_summary.log live under cache_dir so they
    # survive a `rm -rf build-cache/`.
    "$FOUND_BUILD_CONFIG" \
      --fresh \
      -DGPU_TARGETS="$GPU_TARGET" \
      -DCMAKE_CXX_FLAGS="-Rpass-analysis=kernel-resource-usage" 2>&1 |
      tee "$cache_dir/build.log" >&2
    # Under -j>1, multiple TUs write remarks to the log concurrently, so
    # blocks can interleave line-by-line. resource_usage_to_csv.py's parser
    # re-attributes each line by its own "<file>:<line>:" prefix rather than
    # position, so we just pass every remark-key line through unfiltered.
    grep -E '(Function Name|TotalSGPRs|VGPRs|AGPRs|ScratchSize \[bytes/lane\]|Dynamic Stack|Occupancy \[waves/SIMD\]|SGPRs Spill|VGPRs Spill|LDS Size \[bytes/block\]):' \
      "$cache_dir/build.log" >"$cache_dir/resource_usage_summary.log" || true
  )

  git -C "$PROJECT_DIR" worktree remove "$worktree" >&2 || true
  CURRENT_WORKTREE=""

  # Same stdout-capture constraint as above: redirect this script's own
  # report to stderr.
  python3 "$TOOLS_DIR/resource_usage_to_csv.py" \
    --log "$cache_dir/resource_usage_summary.log" \
    --arch "$GPU_TARGET" --build-config "$CONFIG_LABEL" --commit "$sha" \
    --out "$csv" >&2

  # rocshmem_device_bitcode is declared ALL, so the build above already
  # produced it -- measure its whole-program opt -O3 codegen too, not just
  # the production library's cost-gated-LTO numbers. rocSHMEM-specific:
  # silently skipped (note below) for any other project's build.
  local bc_file="$build_dir/librocshmem_device_${GPU_TARGET}.bc"
  local bitcode_csv="$cache_dir/res-${sha}-bitcode.csv"
  if [[ -f "$bc_file" ]]; then
    measure_device_bitcode "$bc_file" "$GPU_TARGET" "$CONFIG_LABEL" "$sha" "$bitcode_csv"
  else
    echo "  [$sha] note: no $bc_file -- skipping device-bitcode resource-usage measurement" >&2
  fi

  echo "$csv"
}

# generate_dashboard <csv1> <csv2> <sha1> <sha2> <diff_csv> <out_html>
# <diff_csv> is any one already-written res_diff_<Column>.csv -- every
# --sort-by choice writes the same row set/columns (only row order differs),
# so one file carries the full multi-metric diff data for the dashboard's
# client-side metric switcher.
generate_dashboard() {
  local csv1="$1" csv2="$2" sha1="$3" sha2="$4" diff_csv="$5" out_html="$6"
  python3 "$TOOLS_DIR/resource_usage_dashboard.py" \
    --baseline-csv "$csv1" --baseline-commit "$sha1" \
    --branch-csv "$csv2" --branch-commit "$sha2" \
    --diff-csv "$diff_csv" --top "$TOP_N" --out "$out_html" >&2
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
# real before/after decision, since AMDGPU LTO codegen isn't deterministic
# build-to-build and a stale baseline can show spurious per-kernel deltas.
if [[ -n "$COMMIT_2" && "$FORCE_REBUILD" == false ]]; then
  echo "  Two-commit comparison: reusing cached builds where available." >&2
  echo "  Pass --force-rebuild for a matched-fresh-pair rebuild before a real" >&2
  echo "  before/after regression decision (AMDGPU LTO codegen is not" >&2
  echo "  build-to-build deterministic)." >&2
fi

echo "=== resource usage: $COMMIT_1${COMMIT_2:+ vs $COMMIT_2} ($GPU_TARGET / $CONFIG_LABEL) ==="

SHA_1="$(git rev-parse --short=12 "$COMMIT_1")"
CSV_1="$(measure_commit "$COMMIT_1" "$SHA_1")"

if [[ -z "$COMMIT_2" ]]; then
  DASHBOARD_1="$(dirname "$CSV_1")/resource_dashboard.html"
  python3 "$TOOLS_DIR/resource_usage_dashboard.py" \
    --baseline-csv "$CSV_1" --baseline-commit "$SHA_1" --top "$TOP_N" --out "$DASHBOARD_1" >&2
  echo ""
  echo "Single-commit snapshot -> $CSV_1"
  echo "  Dashboard -> $DASHBOARD_1"

  BITCODE_CSV_1="${CSV_1%.csv}-bitcode.csv"
  if [[ -f "$BITCODE_CSV_1" ]]; then
    DASHBOARD_1_BITCODE="$(dirname "$CSV_1")/resource_dashboard_bitcode.html"
    python3 "$TOOLS_DIR/resource_usage_dashboard.py" \
      --baseline-csv "$BITCODE_CSV_1" --baseline-commit "$SHA_1" --top "$TOP_N" \
      --out "$DASHBOARD_1_BITCODE" >&2
    echo "  Device bitcode -> $BITCODE_CSV_1"
    echo "  Bitcode dashboard -> $DASHBOARD_1_BITCODE"
  fi
  exit 0
fi

SHA_2="$(git rev-parse --short=12 "$COMMIT_2")"
CSV_2="$(measure_commit "$COMMIT_2" "$SHA_2")"

OUTDIR="${OUTPUT_DIR:-$PARENT_DIR/resource-usage/${GPU_TARGET}-${CONFIG_LABEL}-${SHA_1}-vs-${SHA_2}}"
mkdir -p "$OUTDIR"
cp "$CSV_1" "$OUTDIR/res-${SHA_1}.csv"
cp "$CSV_2" "$OUTDIR/res-${SHA_2}.csv"

for sort_by in "${SORT_BY_TYPE[@]}"; do
  python3 "$TOOLS_DIR/resource_usage_diff.py" \
    --baseline "$CSV_1" \
    --branch "$CSV_2" \
    --out "$OUTDIR/res_diff_${sort_by}.csv" \
    --chart "$OUTDIR/res_diff_${sort_by}.png" \
    --top "$TOP_N" --sort-by "$sort_by" \
    ${MATCH:+--match "$MATCH"}
done
generate_dashboard "$OUTDIR/res-${SHA_1}.csv" "$OUTDIR/res-${SHA_2}.csv" \
  "$SHA_1" "$SHA_2" "$OUTDIR/res_diff_VGPRs.csv" "$OUTDIR/resource_dashboard.html"

# Device-bitcode (whole-program opt -O3) numbers, when both sides have them --
# this inliner has no cost model/per-TU boundary, so a change that's
# safe/beneficial under the production library's cost-gated LTO can still
# regress here.
BITCODE_CSV_1="${CSV_1%.csv}-bitcode.csv"
BITCODE_CSV_2="${CSV_2%.csv}-bitcode.csv"
if [[ -f "$BITCODE_CSV_1" && -f "$BITCODE_CSV_2" ]]; then
  cp "$BITCODE_CSV_1" "$OUTDIR/res-${SHA_1}-bitcode.csv"
  cp "$BITCODE_CSV_2" "$OUTDIR/res-${SHA_2}-bitcode.csv"
  for sort_by in "${SORT_BY_TYPE[@]}"; do
    python3 "$TOOLS_DIR/resource_usage_diff.py" \
      --baseline "$BITCODE_CSV_1" \
      --branch "$BITCODE_CSV_2" \
      --out "$OUTDIR/res_diff_bitcode_${sort_by}.csv" \
      --chart "$OUTDIR/res_diff_bitcode_${sort_by}.png" \
      --top "$TOP_N" --sort-by "$sort_by" \
      ${MATCH:+--match "$MATCH"}
  done
  generate_dashboard "$OUTDIR/res-${SHA_1}-bitcode.csv" "$OUTDIR/res-${SHA_2}-bitcode.csv" \
    "$SHA_1" "$SHA_2" "$OUTDIR/res_diff_bitcode_VGPRs.csv" "$OUTDIR/resource_dashboard_bitcode.html"
else
  echo "  note: device-bitcode resource-usage CSV missing for one or both commits" >&2
  echo "  ($BITCODE_CSV_1, $BITCODE_CSV_2) -- skipping bitcode diff." >&2
fi

echo ""
echo "Done. Self-contained report -> $OUTDIR/"
echo "  Production library: res-${SHA_1}.csv, res-${SHA_2}.csv, res_diff_<Column>.{csv,png}"
echo "  Dashboard:           resource_dashboard.html"
if [[ -f "$OUTDIR/res-${SHA_1}-bitcode.csv" ]]; then
  echo "  Device bitcode:      res-${SHA_1}-bitcode.csv, res-${SHA_2}-bitcode.csv, res_diff_bitcode_<Column>.{csv,png}"
  echo "  Bitcode dashboard:   resource_dashboard_bitcode.html"
fi
echo "  (each report set covers: ${SORT_BY_TYPE[*]})"
