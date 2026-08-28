#!/bin/bash
###############################################################################
# Capture the whole-program AMDGPU LTO inliner's optimization-record remarks
# (`-Xoffload-linker --plugin-opt=opt-remarks-filename=...`) for
# rocshmem_functional_tests, for a single commit or between two commits, and
# turn them into a CSV (lto_inline_remarks_to_csv.py) plus a PNG dashboard:
# a single-commit report (lto_inline_remarks_report.py) or a two-commit diff
# (lto_inline_remarks_diff.py) -- e.g. direct evidence for the "last call to a
# static function" cost bonus investigated in LTO_INLINE_CALLER_COUNT_ISSUE.md.
#
# Each commit is built in an isolated git worktree under /tmp, same as
# resource_usage_compare.sh, so the main working tree is never touched.
#
# Builds here use a SEPARATE build-cache namespace
# ($PROJECTS_DIR/build-cache-lto-remarks) from resource_usage_compare.sh's
# build-cache: this script forces BUILD_TOOLS=OFF, BUILD_UNIT_TESTS=OFF,
# BUILD_EXAMPLES=OFF so rocshmem_functional_tests is the only executable
# target that performs a -fgpu-rdc LTO link -- with more than one
# LTO-linked executable, CMAKE_EXE_LINKER_FLAGS would point every one of them
# at the same opt-remarks-filename, and concurrent parallel-build links would
# clobber/interleave each other's output. Sharing resource_usage_compare.sh's
# build-cache would also mix incompatible cmake configs (different linker
# flags, different BUILD_* options) into the same cache dir.
#
# Usage:
#   ./lto_inline_remarks_compare.sh [OPTIONS]
#
# Options:
#   --commit1 REF         First commit/branch to measure (default: HEAD).
#   --commit2 REF         Second commit/branch to compare against commit1.
#                         Omit to just snapshot commit1 with no diff.
#   --gpu-target ARCH     GPU target architecture (default: gfx950).
#   --build-config CFG    Build config script under scripts/build_configs/
#                         (default: all_backends). Use a lighter config
#                         (e.g. ipc_single) for a faster first look.
#   --force-rebuild       Force a fresh rebuild even if the commit is already
#                         cached (see resource_usage_compare.sh's header for
#                         why this matters for a real before/after decision --
#                         AMDGPU LTO codegen is not build-to-build
#                         deterministic).
#   --top N               Rows shown in the dashboard's per-commit charts and
#                         report panels/CSV sections (default: 20).
#   --output-dir DIR      Where to write the report. Default:
#                         $PROJECTS_DIR/lto-inline-remarks/<gpu>-<config>-<sha1>[-vs-<sha2>]/
#
# Also generates one self-contained interactive HTML dashboard
# (lto_inline_dashboard.html, always on -- no flag needed): a baseline-vs-branch comparison table on top
# (scrollable, filterable) for two-commit runs, plus a per-commit stats/chart/
# table view with an in-page commit selector.
#
# Example:
#   ./lto_inline_remarks_compare.sh --commit1 HEAD~1 --commit2 HEAD \
#     --build-config ipc_single
###############################################################################
set -euo pipefail
shopt -s inherit_errexit

COMMIT_1=""
COMMIT_2=""
GPU_TARGET="gfx950"
BUILD_CONFIG="all_backends"
FORCE_REBUILD=false
TOP_N=20
OUTPUT_DIR=""

_need_arg() {
  if [[ "$2" -lt 2 ]]; then
    echo "ERROR: $1 requires an argument" >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --commit1)       _need_arg "$1" "$#"; COMMIT_1="$2";      shift 2 ;;
    --commit2)       _need_arg "$1" "$#"; COMMIT_2="$2";      shift 2 ;;
    --gpu-target)    _need_arg "$1" "$#"; GPU_TARGET="$2";    shift 2 ;;
    --build-config)  _need_arg "$1" "$#"; BUILD_CONFIG="$2";  shift 2 ;;
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
ROCSHMEM_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECTS_DIR="$(cd "$ROCSHMEM_DIR/.." && pwd)"
cd "$ROCSHMEM_DIR"

TOOLS_DIR="$ROCSHMEM_DIR/scripts/analysis"

CURRENT_WORKTREE=""
_cleanup_worktree() {
  if [[ -n "$CURRENT_WORKTREE" ]]; then
    git -C "$ROCSHMEM_DIR" worktree remove --force "$CURRENT_WORKTREE" 2>/dev/null || true
    CURRENT_WORKTREE=""
  fi
}
trap _cleanup_worktree EXIT

_find_build_config() {
  local worktree="$1" config="$2" result=""
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

# measure_commit <commit> <sha> -> prints the path to that commit's remarks
# CSV (raw .yaml lives alongside it in the same cache dir).
measure_commit() {
  local commit="$1" sha="$2"
  local build_dir="$PROJECTS_DIR/build-cache-lto-remarks/${GPU_TARGET}-${BUILD_CONFIG}-${sha}"
  local cache_dir="$PROJECTS_DIR/lto-inline-remarks/cache/${GPU_TARGET}-${BUILD_CONFIG}-${sha}"
  local yaml="$cache_dir/remarks-${sha}.yaml"
  local csv="$cache_dir/remarks-${sha}.csv"

  if [[ -f "$csv" && "$FORCE_REBUILD" == false ]]; then
    echo "  [$sha] cached -> $csv" >&2
    echo "$csv"
    return
  fi

  echo "  [$sha] building ($GPU_TARGET / $BUILD_CONFIG)..." >&2
  local worktree="/tmp/rocshmem-lto-remarks-${sha}-$$"

  git -C "$ROCSHMEM_DIR" worktree add "$worktree" "$commit" --detach >&2
  CURRENT_WORKTREE="$worktree"

  local FOUND_BUILD_CONFIG
  FOUND_BUILD_CONFIG="$(_find_build_config "$worktree" "$BUILD_CONFIG")"
  if [[ -z "$FOUND_BUILD_CONFIG" ]]; then
    echo "ERROR: Cannot find $BUILD_CONFIG in worktree" >&2
    exit 1
  fi

  rm -rf "$build_dir"
  mkdir -p "$build_dir" "$cache_dir"
  rm -f "$yaml" "$csv"

  (
    cd "$build_dir"
    # The build_config wrapper scripts do `cmake ... $* $src_path` with an
    # UNQUOTED $* -- any multi-word value passed here via -DCMAKE_EXE_LINKER_FLAGS
    # gets re-split back into separate shell words before reaching cmake,
    # so e.g. "-Xoffload-linker --plugin-opt=opt-remarks-with-hotness" arrives
    # as its own top-level CLI argument ("Unknown argument"). CMake seeds
    # CMAKE_EXE_LINKER_FLAGS from the LDFLAGS environment variable on a fresh
    # configure, so set it there instead -- env vars aren't subject to the
    # wrapper's word-splitting. Verified against a scratch cmake project that
    # the full multi-token string survives intact this way.
    export LDFLAGS="-Xoffload-linker --plugin-opt=opt-remarks-filename=${yaml} -Xoffload-linker --plugin-opt=opt-remarks-passes=inline -Xoffload-linker --plugin-opt=opt-remarks-with-hotness"
    "$FOUND_BUILD_CONFIG" \
      --fresh \
      -DGPU_TARGETS="$GPU_TARGET" \
      -DBUILD_TOOLS=OFF \
      -DBUILD_UNIT_TESTS=OFF \
      -DBUILD_EXAMPLES=OFF \
      2>&1 | tee "$cache_dir/build.log" >&2
  )

  git -C "$ROCSHMEM_DIR" worktree remove "$worktree" >&2 || true
  CURRENT_WORKTREE=""

  if [[ ! -s "$yaml" ]]; then
    echo "ERROR: no opt-remarks output at $yaml -- check $cache_dir/build.log" >&2
    exit 1
  fi

  python3 "$TOOLS_DIR/lto_inline_remarks_to_csv.py" --yaml "$yaml" --out "$csv" \
    --arch "$GPU_TARGET" --build-config "$BUILD_CONFIG" --commit "$sha" >&2

  echo "$csv"
}

COMMIT_1="${COMMIT_1:-HEAD}"
SHA_1="$(git rev-parse --short=12 "$COMMIT_1")"

echo "=== LTO inline remarks: $COMMIT_1${COMMIT_2:+ vs $COMMIT_2} ($GPU_TARGET / $BUILD_CONFIG) ==="
CSV_1="$(measure_commit "$COMMIT_1" "$SHA_1")"

if [[ -z "$COMMIT_2" ]]; then
  OUTDIR="${OUTPUT_DIR:-$PROJECTS_DIR/lto-inline-remarks/${GPU_TARGET}-${BUILD_CONFIG}-${SHA_1}}"
  mkdir -p "$OUTDIR"
  cp "$CSV_1" "$OUTDIR/remarks-${SHA_1}.csv"
  python3 "$TOOLS_DIR/lto_inline_remarks_report.py" \
    --csv "$OUTDIR/remarks-${SHA_1}.csv" \
    --out-chart "$OUTDIR/dashboard-${SHA_1}.png" \
    --out-summary "$OUTDIR/summary-${SHA_1}.csv" \
    --top "$TOP_N"
  python3 "$TOOLS_DIR/lto_inline_remarks_dashboard.py" \
    --baseline-csv "$OUTDIR/remarks-${SHA_1}.csv" --baseline-commit "$SHA_1" \
    --top "$TOP_N" --out "$OUTDIR/lto_inline_dashboard.html"
  echo ""
  echo "Single-commit snapshot -> $OUTDIR/"
  echo "  remarks-${SHA_1}.csv, summary-${SHA_1}.csv, dashboard-${SHA_1}.png"
  echo "  Dashboard -> $OUTDIR/lto_inline_dashboard.html"
  echo "  (raw YAML in cache: ${CSV_1%.csv}.yaml)"
  exit 0
fi

SHA_2="$(git rev-parse --short=12 "$COMMIT_2")"
CSV_2="$(measure_commit "$COMMIT_2" "$SHA_2")"

OUTDIR="${OUTPUT_DIR:-$PROJECTS_DIR/lto-inline-remarks/${GPU_TARGET}-${BUILD_CONFIG}-${SHA_1}-vs-${SHA_2}}"
mkdir -p "$OUTDIR"
cp "$CSV_1" "$OUTDIR/remarks-${SHA_1}.csv"
cp "$CSV_2" "$OUTDIR/remarks-${SHA_2}.csv"

python3 "$TOOLS_DIR/lto_inline_remarks_diff.py" \
  --baseline "$OUTDIR/remarks-${SHA_1}.csv" --branch "$OUTDIR/remarks-${SHA_2}.csv" \
  --out "$OUTDIR/remarks_diff_pairs.csv" \
  --summary-out "$OUTDIR/remarks_diff_summary.csv" \
  --chart "$OUTDIR/remarks_diff_dashboard.png" \
  --top "$TOP_N"

python3 "$TOOLS_DIR/lto_inline_remarks_dashboard.py" \
  --baseline-csv "$OUTDIR/remarks-${SHA_1}.csv" --baseline-commit "$SHA_1" \
  --branch-csv "$OUTDIR/remarks-${SHA_2}.csv" --branch-commit "$SHA_2" \
  --pairs-csv "$OUTDIR/remarks_diff_pairs.csv" \
  --top "$TOP_N" --out "$OUTDIR/lto_inline_dashboard.html"

echo ""
echo "Done. Report -> $OUTDIR/"
echo "  remarks-${SHA_1}.csv, remarks-${SHA_2}.csv,"
echo "  remarks_diff_pairs.csv, remarks_diff_summary.csv, remarks_diff_dashboard.png"
echo "  Dashboard -> $OUTDIR/lto_inline_dashboard.html"
