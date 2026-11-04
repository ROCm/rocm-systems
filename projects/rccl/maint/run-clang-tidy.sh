#!/usr/bin/env bash
#
# run-clang-tidy.sh — Run clang-tidy over RCCL's hipified sources.
#
# The enabled checks live in the top-level .clang-tidy, which currently enables
# exactly one. Findings are advisory: this script exits 0 even when clang-tidy
# reports something, unless --strict is given.
#
# Unlike clang-format, clang-tidy needs a parsed translation unit, which means
# a compile_commands.json and hipified sources on disk. That is why this is a
# script rather than a pre-commit hook: pre-commit works from a changed-file
# list and cannot configure a build.
#
# Usage:
#   ./run-clang-tidy.sh                           # analyse RCCL-owned sources
#   ./run-clang-tidy.sh --build-dir DIR           # use an existing build tree
#   ./run-clang-tidy.sh --strict                  # exit non-zero on findings
#   ./run-clang-tidy.sh --include-upstream        # also report NCCL-owned code
#   ./run-clang-tidy.sh src/graph/tuning.cc       # analyse matching files only
#
# By default, findings in files that NVIDIA/nccl also ships at the same path
# are dropped. Those files are maintained upstream, so a finding in one is
# either something to raise with NVIDIA or something the next sync will
# overwrite; either way it is noise when you are trying to see what RCCL owns.
# Filtering needs the `nccl` remote, and the script stops rather than silently
# widening its scope if that remote is missing.
#
# If the build directory does not already have a compile database, the script
# configures one and runs the hipify step, which takes a few minutes but needs
# no GPU and compiles nothing.

set -euo pipefail

MIN_VERSION=18

RCCL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${RCCL_ROOT}/build/clang-tidy"
STRICT=0
FILTER=""
# Any valid architecture does; see the configure step below.
GPU_TARGETS="${GPU_TARGETS:-gfx942}"
INCLUDE_UPSTREAM=0
NCCL_REF="${NCCL_REF:-nccl/master}"

usage() {
  echo "Usage: $0 [--build-dir DIR] [--strict] [--include-upstream] [path-filter]" >&2
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      if [[ $# -lt 2 ]]; then
        usage
        exit 1
      fi
      BUILD_DIR="$2"
      shift
      ;;
    --strict)
      STRICT=1
      ;;
    --include-upstream)
      INCLUDE_UPSTREAM=1
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    --*)
      usage
      exit 1
      ;;
    *)
      if [[ -n "$FILTER" ]]; then
        usage
        exit 1
      fi
      FILTER="$1"
      ;;
  esac
  shift
done

# ---------------------------------------------------------------------------
# Locate clang-tidy and verify version
# ---------------------------------------------------------------------------
# Prefer the clang-tidy that ships with ROCm. A stock upstream clang-tidy is
# built against its own headers and cannot parse ROCm's HIP headers: clang-tidy
# 22.1.8 from PyPI reports 25 clang-diagnostic-errors ("builtin functions must
# be directly called") inside /opt/rocm/include before reaching any RCCL code.
# That is the breakage MIOpen works around with -nogpulib and __builtin_*
# redefinitions; using ROCm's own build means needing none of it.
#
# The LLVM location moved between ROCm releases, hence the candidate list.
CLANG_TIDY_CANDIDATES=(
  "${CLANG_TIDY:-}"
  /opt/rocm/lib/llvm/bin/clang-tidy
  /opt/rocm/llvm/bin/clang-tidy
  clang-tidy
)

CLANG_TIDY=""
for CANDIDATE in "${CLANG_TIDY_CANDIDATES[@]}"; do
  if [[ -n "$CANDIDATE" ]] && command -v "$CANDIDATE" &>/dev/null; then
    CLANG_TIDY="$CANDIDATE"
    break
  fi
done

if [[ -z "$CLANG_TIDY" ]]; then
  echo "ERROR: clang-tidy not found. Looked for ROCm's build under /opt/rocm and" >&2
  echo "       for clang-tidy on PATH. Install clang-tidy >= $MIN_VERSION, or set" >&2
  echo "       CLANG_TIDY to its path." >&2
  exit 2
fi

echo "Using $(command -v "$CLANG_TIDY")"

VERSION=$("$CLANG_TIDY" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 | cut -d. -f1)
if [[ -z "$VERSION" || "$VERSION" -lt "$MIN_VERSION" ]]; then
  echo "ERROR: clang-tidy >= $MIN_VERSION required, found: $("$CLANG_TIDY" --version)" >&2
  exit 2
fi

# LLVM and ROCm ship this driver as `run-clang-tidy`, but the PyPI clang-tidy
# package (which is how ROCm CI jobs install a pinned version) names it
# `run-clang-tidy.py`. Accept either.
CLANG_TIDY_BIN_DIR="$(dirname "$(command -v "$CLANG_TIDY")")"
RUN_CLANG_TIDY=""
for CANDIDATE in "${CLANG_TIDY_BIN_DIR}/run-clang-tidy" "${CLANG_TIDY_BIN_DIR}/run-clang-tidy.py"; do
  if [[ -x "$CANDIDATE" ]]; then
    RUN_CLANG_TIDY="$CANDIDATE"
    break
  fi
done

if [[ -z "$RUN_CLANG_TIDY" ]]; then
  echo "ERROR: neither run-clang-tidy nor run-clang-tidy.py found in ${CLANG_TIDY_BIN_DIR}." >&2
  exit 2
fi

# run-clang-tidy defaults to whatever `clang-tidy` is on PATH, which is not
# necessarily the one we just version-checked.
RUN_CLANG_TIDY_ARGS=(-clang-tidy-binary "$(command -v "$CLANG_TIDY")")

# ---------------------------------------------------------------------------
# Make sure we have a compile database and hipified sources
# ---------------------------------------------------------------------------
if [[ ! -f "${BUILD_DIR}/compile_commands.json" ]]; then
  echo "No compile database in ${BUILD_DIR}; configuring (no GPU needed, compiles nothing)..."
  # GPU_TARGETS must be passed as a cache variable; CMake does not read it
  # from the environment, and left unset the configure falls back to probing
  # every architecture in DEFAULT_GPUS. One is enough: hipify stages host
  # sources and compiles no device code, so the arch only has to be valid, and
  # pinning it also stops the configure probing for a local GPU on machines
  # that have none.
  cmake \
    -S "$RCCL_ROOT" \
    -B "$BUILD_DIR" \
    -DBUILD_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DGPU_TARGETS="$GPU_TARGETS"
  cmake --build "$BUILD_DIR" --target hipify_all -j "$(nproc)"
  # csv_tuner.cc includes a header generated by this target. Without it,
  # clang-tidy reports a clang-diagnostic-error for that one file instead of
  # analysing it.
  cmake --build "$BUILD_DIR" --target rccl_embedded_tuner_configs -j "$(nproc)"
fi

# ---------------------------------------------------------------------------
# Work out which files NCCL owns, so their findings can be dropped
# ---------------------------------------------------------------------------
UPSTREAM_LIST=""
if [[ "$INCLUDE_UPSTREAM" -eq 0 ]]; then
  if ! git -C "$RCCL_ROOT" rev-parse --verify --quiet "$NCCL_REF" >/dev/null; then
    echo "ERROR: cannot resolve '$NCCL_REF', needed to tell RCCL-owned files from" >&2
    echo "       NCCL-owned ones. Add the upstream remote (fetch-only):" >&2
    echo "" >&2
    echo "         git remote add nccl git@github.com:NVIDIA/nccl.git" >&2
    echo "         git remote set-url --push nccl no_push" >&2
    echo "         git fetch nccl" >&2
    echo "" >&2
    echo "       Or pass --include-upstream to report on every file." >&2
    exit 2
  fi

  UPSTREAM_LIST="$(mktemp)"
  # shellcheck disable=SC2064
  trap "rm -f '$UPSTREAM_LIST'" EXIT
  # --full-tree matters: without it git ls-tree reports paths relative to the
  # current directory and restricts itself to that subtree, and this script
  # runs from projects/rccl, which does not exist in NCCL's tree. The result
  # would be an empty list and silently unfiltered output.
  git -C "$RCCL_ROOT" ls-tree -r --full-tree --name-only "$NCCL_REF" \
    | grep -E '\.(c|cc|cpp|h|hpp|cu|cuh)$' \
    | sed 's|^|/|' \
    | sort -u > "$UPSTREAM_LIST"
fi

# ---------------------------------------------------------------------------
# Analyse
#
# clang-tidy sees the hipified copies under ${BUILD_DIR}/hipify, whose paths
# and line numbers correspond to the originals under src/. Rewrite the paths
# back so the output names files you can actually open.
# ---------------------------------------------------------------------------
# Pass the config explicitly. clang-tidy finds .clang-tidy by walking up from
# each analysed file, and the files it analyses are the hipified copies in the
# build directory, which may be outside the source tree entirely. Relying on
# the walk-up silently yields zero findings.
ARGS=(-p "$BUILD_DIR" -quiet -j "$(nproc)" -config-file "${RCCL_ROOT}/.clang-tidy"
  "${RUN_CLANG_TIDY_ARGS[@]}")
if [[ -n "$FILTER" ]]; then
  ARGS+=("$FILTER")
fi

set +e
"$RUN_CLANG_TIDY" "${ARGS[@]}" 2>/dev/null \
  | grep -E '(warning|error):' \
  | sed "s|${BUILD_DIR}/hipify/||" \
  | sed 's/_tmp\.cc:/.cc:/' \
  | sort -u > "${BUILD_DIR}/clang-tidy-all.txt"
set -e

if [[ -n "$UPSTREAM_LIST" ]]; then
  # Match on the file part of each finding, anchored with a leading slash and a
  # trailing colon so that, say, src/init.cc does not also match
  # src/transport/init.cc.
  grep -vFf <(sed 's|$|:|' "$UPSTREAM_LIST") \
    <(sed 's|^|/|' "${BUILD_DIR}/clang-tidy-all.txt") \
    | sed 's|^/||' > "${BUILD_DIR}/clang-tidy-findings.txt" || true
else
  cp "${BUILD_DIR}/clang-tidy-all.txt" "${BUILD_DIR}/clang-tidy-findings.txt"
fi

cat "${BUILD_DIR}/clang-tidy-findings.txt"

COUNT=$(wc -l < "${BUILD_DIR}/clang-tidy-findings.txt")
TOTAL=$(wc -l < "${BUILD_DIR}/clang-tidy-all.txt")
echo
if [[ -n "$UPSTREAM_LIST" ]]; then
  echo "clang-tidy: ${COUNT} finding(s) in RCCL-owned files" \
    "($((TOTAL - COUNT)) more in NCCL-owned files, hidden; --include-upstream shows them)."
else
  echo "clang-tidy: ${COUNT} finding(s), RCCL-owned and NCCL-owned alike."
fi
echo "Full list: ${BUILD_DIR}/clang-tidy-findings.txt"

if [[ "$STRICT" -eq 1 && "$COUNT" -gt 0 ]]; then
  exit 1
fi

exit 0
