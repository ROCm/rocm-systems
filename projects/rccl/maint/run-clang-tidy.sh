#!/usr/bin/env bash
#
# run-clang-tidy.sh — Run clang-tidy over RCCL's hipified sources.
#
# The enabled checks live in projects/rccl/.clang-tidy, which currently enables
# exactly one. Findings are advisory: this script exits 0 even when clang-tidy
# reports something, unless --strict is given.
#
# Unlike clang-format, clang-tidy needs a parsed translation unit, which means
# a compile_commands.json and hipified sources on disk. That is why it is a
# manual-stage pre-commit hook rather than an ordinary one: pre-commit works
# from a changed-file list and the build step is far too slow for `git commit`.
# The hook runs this script, so the two cannot drift.
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
# Prefer ROCm's clang-tidy: a stock upstream build cannot parse ROCm's HIP
# headers and reports 25 errors inside /opt/rocm/include before reaching any
# RCCL code. Two paths, because the LLVM location moved between releases.
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

# LLVM ships this driver as `run-clang-tidy`, the PyPI package as
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

# Pin the binary: run-clang-tidy otherwise takes whatever is first on PATH.
RUN_CLANG_TIDY_ARGS=(-clang-tidy-binary "$(command -v "$CLANG_TIDY")")

# ---------------------------------------------------------------------------
# Make sure we have a compile database and hipified sources
# ---------------------------------------------------------------------------
if [[ ! -f "${BUILD_DIR}/compile_commands.json" ]]; then
  echo "No compile database in ${BUILD_DIR}; configuring (no GPU needed, compiles nothing)..."
  # GPU_TARGETS must be a cache variable; CMake ignores the environment, and
  # unset it probes every arch in DEFAULT_GPUS. Any valid arch does, since
  # nothing device-side is compiled.
  cmake \
    -S "$RCCL_ROOT" \
    -B "$BUILD_DIR" \
    -DBUILD_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DGPU_TARGETS="$GPU_TARGETS"
fi

# Every run: skipping this analyses the previous run's hipified copies and
# misses any edit since. Incremental, ~2s when nothing changed. The tuner
# target generates a header csv_tuner.cc includes.
cmake --build "$BUILD_DIR" \
  --target hipify_all rccl_embedded_tuner_configs \
  -j "$(nproc)"

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
  # --full-tree matters: otherwise ls-tree restricts itself to the current
  # directory, which does not exist in NCCL's tree, giving an empty list and
  # silently unfiltered output.
  git -C "$RCCL_ROOT" ls-tree -r --full-tree --name-only "$NCCL_REF" \
    | grep -E '\.(c|cc|cpp|h|hpp|cu|cuh)$' \
    | sed 's|^|/|' \
    | sort -u > "$UPSTREAM_LIST"
fi

# ---------------------------------------------------------------------------
# Analyse
# ---------------------------------------------------------------------------
# Explicit -config-file: clang-tidy finds .clang-tidy by walking up from each
# analysed file, and those are the hipified copies in the build directory,
# possibly outside the tree. Relying on the walk-up silently yields nothing.
ARGS=(-p "$BUILD_DIR" -quiet -j "$(nproc)" -config-file "${RCCL_ROOT}/.clang-tidy"
  "${RUN_CLANG_TIDY_ARGS[@]}")
if [[ -n "$FILTER" ]]; then
  ARGS+=("$FILTER")
fi

# Keep stderr and check the driver's own status via PIPESTATUS: a driver that
# dies early produces no output, which otherwise reads as a clean run.
STDERR_LOG="${BUILD_DIR}/clang-tidy-stderr.txt"
set +e
"$RUN_CLANG_TIDY" "${ARGS[@]}" 2>"$STDERR_LOG" \
  | grep -E '(warning|error):' \
  | sed "s|${BUILD_DIR}/hipify/||" \
  | sed 's/_tmp\.cc:/.cc:/' \
  | sort -u > "${BUILD_DIR}/clang-tidy-all.txt"
DRIVER_STATUS=${PIPESTATUS[0]}
set -e

# A non-zero status means a translation unit failed to parse, not that there
# were findings: the enabled checks are plain warnings and .clang-tidy sets no
# WarningsAsErrors, so findings alone exit 0.
#
# Count the parse errors here, before the NCCL filter, which would otherwise
# drop the only evidence of one in an NCCL-owned file and leave a broken run
# looking like a clean one.
PARSE_ERRORS=$(grep -cE 'error:' "${BUILD_DIR}/clang-tidy-all.txt" || true)

if [[ "$PARSE_ERRORS" -gt 0 ]]; then
  echo "ERROR: ${PARSE_ERRORS} compiler error(s); the analysis did not see every" >&2
  echo "       translation unit, so a clean finding list would be misleading." >&2
  echo "       First 20:" >&2
  grep -E 'error:' "${BUILD_DIR}/clang-tidy-all.txt" | head -20 >&2
  exit 2
fi

if [[ "$DRIVER_STATUS" -ne 0 && ! -s "${BUILD_DIR}/clang-tidy-all.txt" ]]; then
  echo "ERROR: run-clang-tidy exited ${DRIVER_STATUS} and produced no findings;" >&2
  echo "       this is a failure to run, not a clean result. stderr:" >&2
  tail -20 "$STDERR_LOG" >&2
  exit 2
fi

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
