#!/bin/bash
###############################################################################
# build_and_test.sh -- build rocshmem4py against an installed rocSHMEM and,
# optionally, run its test suite.
#
# rocshmem4py is a standalone Python project that builds on top of a
# pre-installed rocSHMEM C++ library. This script wires up the standard flow:
#
#     point at a rocSHMEM install  ->  build the extension  ->  pytest
#
# rocSHMEM is discovered through its exported CMake package via
# CMAKE_PREFIX_PATH (find_package(rocshmem CONFIG)); ROCSHMEM_HOME is accepted
# as a convenience alias. This script does NOT build rocSHMEM itself -- build
# and install the C++ library first (see the project README).
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print() { echo "[build_and_test] $1"; }
fail()  { echo "[ERROR] $1" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage: ./build_and_test.sh [--prefix <rocshmem-install>] [OPTIONS]

Options:
  -p, --prefix DIR    rocSHMEM install prefix (has lib/cmake/rocshmem/).
                      Defaults to CMAKE_PREFIX_PATH or ROCSHMEM_HOME.
      --rocm-path DIR ROCm install directory (default: $ROCM_PATH or /opt/rocm).
  -n, --nprocs N      Number of PEs for the test run (default: 2).
  -l, --launcher L    torchrun (default, IPC/GDA) | mpirun (RO backend).
      --wheel         Build a distributable wheel (python -m build) instead of
                      an editable install; implies --build-only.
      --build-only    Build the extension but do not run tests.
      --no-isolation  Pass --no-build-isolation to pip (use preinstalled
                      nanobind/cmake instead of fetching them).
  -h, --help          Show this help.

Examples:
  ./build_and_test.sh --prefix /opt/rocshmem -l torchrun -n 2
  CMAKE_PREFIX_PATH=/opt/rocshmem ./build_and_test.sh --build-only
  ./build_and_test.sh --prefix /opt/rocshmem --wheel
EOF
}

# Defaults
PREFIX="${CMAKE_PREFIX_PATH:-${ROCSHMEM_HOME:-}}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
NUM_PROCS=2
LAUNCHER="${ROCSHMEM_LAUNCHER:-torchrun}"
BUILD_ONLY=false
BUILD_WHEEL=false
PIP_ISOLATION=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--prefix)    PREFIX="$2"; shift 2 ;;
    --rocm-path)    ROCM_PATH="$2"; shift 2 ;;
    -n|--nprocs)    NUM_PROCS="$2"; shift 2 ;;
    -l|--launcher)  LAUNCHER="$2"; shift 2 ;;
    --wheel)        BUILD_WHEEL=true; BUILD_ONLY=true; shift ;;
    --build-only)   BUILD_ONLY=true; shift ;;
    --no-isolation) PIP_ISOLATION="--no-build-isolation"; shift ;;
    -h|--help)      usage; exit 0 ;;
    *)              fail "Unknown argument: $1 (see --help)" ;;
  esac
done

[[ -n "$PREFIX" ]] || fail "No rocSHMEM install prefix. Pass --prefix <dir> or set CMAKE_PREFIX_PATH/ROCSHMEM_HOME."
# Only the first path entry is validated for the config package; CMAKE_PREFIX_PATH
# may legitimately be a colon-separated list.
_first_prefix="${PREFIX%%:*}"
[[ -f "$_first_prefix/lib/cmake/rocshmem/rocshmem-config.cmake" ]] || \
  fail "No rocSHMEM CMake package under '$_first_prefix/lib/cmake/rocshmem'. Is that a rocSHMEM install prefix?"

export CMAKE_PREFIX_PATH="$PREFIX"
export ROCM_PATH
# The extension statically links librocshmem.a but still needs ROCm runtime
# libs (and any rocSHMEM shared deps) on the loader path at import/test time.
export LD_LIBRARY_PATH="$_first_prefix/lib:$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
# launch_test.sh seeds its LD_LIBRARY_PATH from ROCSHMEM_BUILD.
export ROCSHMEM_BUILD="$_first_prefix/lib"

print "rocSHMEM install : $_first_prefix"
print "ROCm path        : $ROCM_PATH"

if [[ "$BUILD_WHEEL" == true ]]; then
  # `python -m build` spells the isolation opt-out differently than pip.
  build_isolation=""
  [[ -n "$PIP_ISOLATION" ]] && build_isolation="--no-isolation"
  print "Building distributable wheel (python -m build --wheel)..."
  python -m build --wheel $build_isolation .
  print "Wheel written to $SCRIPT_DIR/dist/."
  exit 0
fi

print "Building rocshmem4py extension (editable install)..."
pip install $PIP_ISOLATION -e .

if [[ "$BUILD_ONLY" == true ]]; then
  print "Build complete (--build-only); skipping tests."
  exit 0
fi

print "Running test suite ($LAUNCHER, n=$NUM_PROCS)..."
exec ./launch_test.sh -l "$LAUNCHER" -n "$NUM_PROCS" -c "pytest tests/ -v"
