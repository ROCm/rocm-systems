#!/usr/bin/env bash
################################################################################
# run_gfx1250_test.sh - End-to-end validation for the hotswap transpiler's
#                       opcode-map / raiser stack on the gfx1250 corpus.
#
# Regression bar:
#   batch_raise_test against hotswap/test_data/gfx1250/ must achieve 100%
#   kernel lift success. Any failure exits non-zero.
#
# Optional GPU pass:
#   If HIP is found at configure time and a GPU is visible, also runs
#   gfx1250_gpu_test (gfx1250 -> gfx942 lowering, executed on-device).
#
# Usage:
#   ./run_gfx1250_test.sh            # configure + build + run
#   ./run_gfx1250_test.sh --no-gpu   # skip gfx1250_gpu_test even if HIP found
#   LLVM_DIR=/path/to/llvm/build/lib/cmake/llvm ./run_gfx1250_test.sh
#
# Requires:
#   - An LLVM *build tree* (not install tree) with AMDGPU target enabled.
#   - cmake, ninja, g++.
#   - (Optional) /opt/rocm-*/lib/cmake/hip for GPU tests.
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
TEST_DATA="${SCRIPT_DIR}/../test_data/gfx1250"

: "${LLVM_DIR:=${HOME}/shared-llvm/lib/cmake/llvm}"
: "${ROCM_DIR:=/opt/rocm-7.2.1}"

RUN_GPU=1
for arg in "$@"; do
  case "$arg" in
    --no-gpu) RUN_GPU=0 ;;
    -h|--help)
      sed -n '2,25p' "$0"
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument '$arg'" >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$LLVM_DIR" ]]; then
  echo "ERROR: LLVM_DIR='$LLVM_DIR' does not exist." >&2
  echo "Point LLVM_DIR at <llvm-build>/lib/cmake/llvm (build tree, not install)." >&2
  exit 2
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

CMAKE_ARGS=(
  -G Ninja
  -DCMAKE_CXX_COMPILER=g++
  -DLLVM_DIR="$LLVM_DIR"
)
if [[ -d "$ROCM_DIR" ]]; then
  CMAKE_ARGS+=(
    -DCMAKE_PREFIX_PATH="${ROCM_DIR};${LLVM_DIR%/lib/cmake/llvm}"
    -Dhip_DIR="${ROCM_DIR}/lib/cmake/hip"
  )
fi

echo "=== configure ==="
cmake .. "${CMAKE_ARGS[@]}"

echo "=== build batch_raise_test ==="
ninja batch_raise_test

echo "=== batch_raise_test (gfx1250 corpus) ==="
./batch_raise_test "$TEST_DATA" --isa=gfx1250
echo "batch_raise_test: PASS"

if [[ "$RUN_GPU" == "1" && -f build.ninja ]] && \
   ninja -t query gfx1250_gpu_test >/dev/null 2>&1; then
  echo "=== build gfx1250_gpu_test ==="
  ninja gfx1250_gpu_test
  if command -v rocm-smi >/dev/null 2>&1 && rocm-smi -i >/dev/null 2>&1; then
    echo "=== gfx1250_gpu_test (on-device) ==="
    ./gfx1250_gpu_test
    echo "gfx1250_gpu_test: PASS"
  else
    echo "=== gfx1250_gpu_test: built but skipped (no GPU visible) ==="
  fi
else
  echo "=== gfx1250_gpu_test: skipped (HIP not configured or --no-gpu) ==="
fi

echo
echo "All enabled validations passed."
