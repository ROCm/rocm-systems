#!/bin/bash
###############################################################################
# Compare per-kernel GPU resource usage (VGPR/SGPR/AGPR/scratch/LDS/occupancy)
# between two commits (or one commit vs the current working tree).
#
# AMDGPU LTO codegen (register allocation, instruction scheduling, internal-
# linkage symbol layout) is measurably NON-DETERMINISTIC build-to-build, even
# for byte-identical, unmodified source. Because of this, a real two-commit
# comparison (COMMIT_2 set) always rebuilds BOTH sides fresh, back-to-back, in
# this one invocation ("matched-fresh-pair") -- diffing a freshly-built branch
# against a baseline built at some earlier, unrelated point in time (e.g. a
# stale cached CSV from a previous run) can show spurious per-kernel deltas
# that are pure build noise, not a consequence of the change under test.
#
# Builds are still cached per (gpu_target, build_config, commit) under
# $PROJECTS_DIR/build-cache, but that cache is only trusted for a
# single-commit snapshot (COMMIT_2 unset) or when --skip-build is passed
# explicitly (e.g. fast iteration on report/chart formatting, or checking a
# new commit against an already-measured baseline, where the absolute
# numbers don't matter). Any real before/after decision should not use
# --skip-build for a two-commit comparison. The durable outputs of a build
# (res-<sha>.csv, build.log, resource_usage_summary.log) live separately under
# $PROJECTS_DIR/resource-usage/cache, so build-cache can be wiped for disk
# space without losing prior measurements.
#
# Each commit is built in an isolated git worktree under /tmp so the main
# working tree is never touched — uncommitted changes are safe.
#
# `rocshmem_device_bitcode` (DeviceBitcode.cmake's librocshmem_device_<arch>.bc,
# consumed by rocshmem_hipmodule_init/Triton-PyTorch JIT) is declared ALL, so
# the same matched-fresh-pair build above already produces it as a side
# effect. Its final `opt -O3` is an ungated whole-program inliner, a
# materially different regime from the production library's cost-gated LTO --
# this script also backend-compiles that .bc with
# -Rpass-analysis=kernel-resource-usage (see measure_device_bitcode() below)
# and diffs it separately (res_diff_bitcode_<Column>.{csv,png}), so a change
# that looks safe under the production library alone doesn't get treated as
# validated everywhere.
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
#   --skip-build          Opt out of matched-fresh-pair rebuilding for a
#                         two-commit comparison: reuse each commit's cached
#                         CSV if one exists, and only build the commit(s)
#                         that aren't cached yet, instead of forcing both
#                         sides to rebuild fresh. Useful for fast iteration
#                         on report/chart formatting (when both sides are
#                         already cached, nothing gets built) or for
#                         checking a new commit against an
#                         already-measured baseline without repaying to
#                         rebuild it. NOT for a real before/after
#                         performance or regression decision, since it can
#                         reintroduce build-to-build LTO noise as a
#                         confound if either side reuses a build from a
#                         different point in time.
#   --force-rebuild       Rebuild+re-extract even if the commit is already
#                         cached (needed after changing --build-config or the
#                         resource-usage extraction scripts themselves). This
#                         is now the default for two-commit comparisons; the
#                         flag remains for single-commit snapshots and as a
#                         no-op for explicitness.
#   --match REGEX         Pin kernels matching this regex (against demangled or
#                         mangled name, case-insensitive) to the top of every
#                         report/chart regardless of delta.
#   --output-dir DIR      Directory to write the comparison report (CSVs +
#                         charts) to. Default:
#                         $PROJECTS_DIR/resource-usage/<gpu>-<config>-<sha1>-vs-<sha2>/
#
# Example: compare two explicit commits (both rebuilt fresh, matched-pair)
#   ./resource_usage_compare.sh --commit1 d48c64f6e --commit2 3caf8d080 \
#     --build-config all_backends
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
    --skip-build)    SKIP_BUILD=true;    shift ;;
    --force-rebuild) FORCE_REBUILD=true; shift ;;
    --match)         _need_arg "$1" "$#"; MATCH="$2";         shift 2 ;;
    --output-dir)    _need_arg "$1" "$#"; OUTPUT_DIR="$2";    shift 2 ;;
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

# Must match resource_usage_diff.py's NUMERIC_COLS -- these are the valid
# --sort-by values. Also duplicated in .claude/skills/rocshmem-resource-usage/
# scripts/compare.sh; keep both lists in sync.
SORT_BY_TYPE=(
  "VGPRs" "TotalSGPRs" "AGPRs" "ScratchBytesPerLane"
  "OccupancyWavesPerSIMD" "SGPRsSpill" "VGPRsSpill" "LDSBytesPerBlock"
)

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

# Backend-compile a device-bitcode artifact (DeviceBitcode.cmake's
# librocshmem_device_<arch>.bc, produced by the `rocshmem_device_bitcode`
# target -- ALL, so already built by the time this is called) directly with
# -Rpass-analysis=kernel-resource-usage to get its resource-usage remarks.
# This is not a re-optimization: DeviceBitcode.cmake's own pipeline (clang
# -emit-llvm, llvm-link, opt -O3) never runs AMDGPU instruction selection, so
# it can never emit these remarks itself -- that codegen normally only
# happens later, at JIT time (rocshmem_hipmodule_init). Backend-compiling the
# .bc here just reuses those same codegen decisions early, for reporting.
#
# The .bc has no debug info (llvm-link merges many TUs, opt -O3 runs with no
# -g), so source_file/line in the output CSV are the placeholder
# "<bitcode>"/0 for every row -- only mangled_name (and the resource columns)
# are meaningful, which is fine since resource_usage_diff.py keys/compares by
# (arch, build_config, mangled_name).
measure_device_bitcode() {
  local bc_file="$1" arch="$2" build_config="$3" commit="$4" out_csv="$5"

  # Same search order as DeviceBitcode.cmake's find_program(LLVM_CLANG ...)
  # (plus ROCM_HOME), so this backend-compile uses the same ROCm clang++ that
  # built the .bc, falling back to PATH only if none of those are set.
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
  workdir="$(mktemp -d /tmp/rocshmem-device-bitcode-XXXXXX)"
  # A RETURN trap isn't scoped to this function -- left registered, it also
  # fires on the *caller's* next return, by which point $workdir is out of
  # scope and set -u aborts the script ("workdir: unbound variable"). Clear
  # it as part of firing so it only ever runs once, for this call.
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

  # Normalize `remark: <unknown>:0:0: Function Name: X [-Rpass-analysis=...]`
  # (no frontend source-location metadata on this direct backend-only
  # invocation) into the `<file>:<line>:<col>: Key: value` shape
  # resource_usage_to_csv.py expects.
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
  local build_dir="$PROJECTS_DIR/build-cache/${GPU_TARGET}-${BUILD_CONFIG}-${sha}"
  local cache_dir="$PROJECTS_DIR/resource-usage/cache/${GPU_TARGET}-${BUILD_CONFIG}-${sha}"
  local csv="$cache_dir/res-${sha}.csv"

  if [[ -f "$csv" && "$FORCE_REBUILD" == false ]]; then
    echo "  [$sha] cached -> $csv" >&2
    echo "$csv"
    return
  fi

  echo "  [$sha] building ($GPU_TARGET / $BUILD_CONFIG)..." >&2
  local worktree="/tmp/rocshmem-resource-usage-${sha}-$$"

  git -C "$ROCSHMEM_DIR" worktree add "$worktree" "$commit" --detach >&2
  CURRENT_WORKTREE="$worktree"

  local FOUND_BUILD_CONFIG
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
  mkdir -p "$cache_dir"
  # resource_usage_to_csv.py merges new rows into any pre-existing --out file
  # (by design, for incremental single-build runs) -- but that means a stale
  # CSV left over from a previous measure_commit() run of this same commit
  # would silently combine with this fresh build's rows instead of being
  # replaced, since they key on (arch, build_config, source_file, line,
  # mangled_name) and each build's ephemeral worktree path differs. That
  # breaks the "matched-fresh-pair" guarantee documented at the top of this
  # script. Since we're about to rebuild unconditionally at this point in the
  # function, always start this commit's CSVs from a clean slate.
  rm -f "$csv" "$cache_dir/res-${sha}-bitcode.csv"
  (
    cd "$build_dir"
    # measure_commit's own stdout is captured by the caller ($(measure_commit ...)) and
    # must contain only the final `echo "$csv"` path below -- tee's stdout copy of the
    # build log must go to stderr (>&2), not stdout, or it corrupts the captured path
    # (and can make it megabytes long, blowing out ARG_MAX in later `cp "$CSV_1" ...`).
    # build.log/resource_usage_summary.log are written under cache_dir (not
    # build_dir) so they survive a `rm -rf build-cache/`.
    "$FOUND_BUILD_CONFIG" \
      --fresh \
      -DGPU_TARGETS="$GPU_TARGET" \
      -DCMAKE_CXX_FLAGS="-Rpass-analysis=kernel-resource-usage" 2>&1 |
      tee "$cache_dir/build.log" >&2
    # A plain "-A9" context window assumes each kernel's ~9-line remark block
    # stays contiguous in the log, which only holds for a serial (-j1) build.
    # Under the normal -j>1 build, multiple TUs' compiler processes emit their
    # remarks to this shared log concurrently -- each line is written
    # atomically, but two kernels' blocks can interleave line-by-line, so a
    # fixed window after one kernel's "Function Name:" line can both pull in
    # a neighboring kernel's lines and cut off that kernel's own later lines.
    # resource_usage_to_csv.py's parser re-attributes every line by its own
    # "<file>:<line>:" prefix (immune to interleaving) instead of by position,
    # so just pass through every remark-key line unfiltered by position.
    grep -E '(Function Name|TotalSGPRs|VGPRs|AGPRs|ScratchSize \[bytes/lane\]|Dynamic Stack|Occupancy \[waves/SIMD\]|SGPRs Spill|VGPRs Spill|LDS Size \[bytes/block\]):' \
      "$cache_dir/build.log" >"$cache_dir/resource_usage_summary.log" || true
  )

  git -C "$ROCSHMEM_DIR" worktree remove "$worktree" >&2 || true
  CURRENT_WORKTREE=""

  # measure_commit's stdout is captured by the caller (CSV_1="$(measure_commit ...)")
  # and must contain only the final `echo "$csv"` path -- redirect this script's own
  # report (which prints to stdout) to stderr so it stays visible without corrupting
  # the captured path.
  python3 "$TOOLS_DIR/resource_usage_to_csv.py" \
    --log "$cache_dir/resource_usage_summary.log" \
    --arch "$GPU_TARGET" --build-config "$BUILD_CONFIG" --commit "$sha" \
    --out "$csv" >&2

  # `rocshmem_device_bitcode` is declared ALL in DeviceBitcode.cmake, so the
  # plain `cmake --build .` above already produced it as a side effect --
  # measure the device-bitcode artifact's whole-program `opt -O3` codegen too
  # (a materially different, ungated inlining regime from the production
  # library's cost-gated LTO) rather than only ever checking the
  # production-library numbers.
  local bc_file="$build_dir/librocshmem_device_${GPU_TARGET}.bc"
  local bitcode_csv="$cache_dir/res-${sha}-bitcode.csv"
  if [[ -f "$bc_file" ]]; then
    measure_device_bitcode "$bc_file" "$GPU_TARGET" "$BUILD_CONFIG" "$sha" "$bitcode_csv"
  else
    echo "  [$sha] note: no $bc_file -- skipping device-bitcode resource-usage measurement" >&2
  fi

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

# Matched-fresh-pair discipline: a real two-commit comparison must not diff a
# freshly-built branch against a baseline built at some earlier, unrelated
# point in time -- AMDGPU LTO codegen is not deterministic build-to-build, so
# that comparison can show spurious per-kernel deltas that are pure build
# noise rather than a consequence of the change under test. Force both sides
# to rebuild fresh in this invocation unless the caller explicitly opted out
# (--skip-build).
if [[ -n "$COMMIT_2" && "$SKIP_BUILD" == false ]]; then
  if [[ "$FORCE_REBUILD" == false ]]; then
    echo "  Two-commit comparison: forcing a fresh matched-pair rebuild of both" >&2
    echo "  commits (pass --skip-build to reuse cached builds instead -- not" >&2
    echo "  recommended for a real before/after decision)." >&2
  fi
  FORCE_REBUILD=true
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

OUTDIR="${OUTPUT_DIR:-$PROJECTS_DIR/resource-usage/${GPU_TARGET}-${BUILD_CONFIG}-${SHA_1}-vs-${SHA_2}}"
mkdir -p "$OUTDIR"
cp "$CSV_1" "$OUTDIR/res-${SHA_1}.csv"
cp "$CSV_2" "$OUTDIR/res-${SHA_2}.csv"

for sort_by in "${SORT_BY_TYPE[@]}"; do
  python3 "$TOOLS_DIR/resource_usage_diff.py" \
    --baseline "$CSV_1" \
    --branch "$CSV_2" \
    --out "$OUTDIR/res_diff_${sort_by}.csv" \
    --chart "$OUTDIR/res_diff_${sort_by}.png" \
    --top 20 --sort-by "$sort_by" \
    ${MATCH:+--match "$MATCH"}
done

# Device-bitcode artifact (DeviceBitcode.cmake's whole-program `opt -O3`)
# numbers, when both sides have them -- see the note in measure_commit(). A
# change that looks safe/beneficial under the production library's
# cost-gated LTO can still regress here, since this inliner has no cost
# model and no per-TU boundary.
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
      --top 20 --sort-by "$sort_by" \
      ${MATCH:+--match "$MATCH"}
  done
else
  echo "  note: device-bitcode resource-usage CSV missing for one or both commits" >&2
  echo "  ($BITCODE_CSV_1, $BITCODE_CSV_2) -- skipping bitcode diff." >&2
fi

echo ""
echo "Done. Self-contained report -> $OUTDIR/"
echo "  Production library: res-${SHA_1}.csv, res-${SHA_2}.csv, res_diff_<Column>.{csv,png}"
if [[ -f "$OUTDIR/res-${SHA_1}-bitcode.csv" ]]; then
  echo "  Device bitcode:      res-${SHA_1}-bitcode.csv, res-${SHA_2}-bitcode.csv, res_diff_bitcode_<Column>.{csv,png}"
fi
echo "  (each report set covers: ${SORT_BY_TYPE[*]})"
