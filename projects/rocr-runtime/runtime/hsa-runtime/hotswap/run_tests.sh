#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
ROCM_PREFIX="${ROCM_PREFIX:-/opt/rocm}"

if [ -z "${LLVM_DIR:-}" ]; then
  llvm_cmake_dir=$(find -L "${ROCM_PREFIX}" -name LLVMConfig.cmake -printf '%h\n' 2>/dev/null | head -1)
  if [ -z "$llvm_cmake_dir" ]; then
    echo "ERROR: Cannot find LLVMConfig.cmake under ${ROCM_PREFIX}" >&2
    echo "Install rocm-llvm-dev or set LLVM_DIR." >&2
    exit 1
  fi
  LLVM_DIR="$llvm_cmake_dir"
fi

NO_GPU=0
TIER1_PASSED=0
TIER1_FAILED=0
TIER2_PASSED=0
TIER2_FAILED=0

for arg in "$@"; do
  case "$arg" in
    --no-gpu) NO_GPU=1 ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

run_test() {
  local tier="$1" name="$2"
  shift 2
  echo "--- [$tier] $name ---"
  if "$@"; then
    echo "    PASS"
    return 0
  else
    echo "    FAIL (exit $?)"
    return 1
  fi
}

# ── Build ─────────────────────────────────────────────────────────────────────

echo "========================================="
echo "  HotSwap Test Suite"
echo "========================================="
echo ""
echo "Building hotswap library + tests..."

MLIR_DIR=$(find -L "${ROCM_PREFIX}" -name MLIRConfig.cmake -printf '%h\n' 2>/dev/null | head -1)

cmake -B "${BUILD_DIR}" -S "${SCRIPT_DIR}" -GNinja \
  -DLLVM_DIR="${LLVM_DIR}" \
  -DMLIR_DIR="${MLIR_DIR}" \
  -DCMAKE_BUILD_TYPE=Release

ninja -C "${BUILD_DIR}"

echo ""

# ── Tier 1: No GPU required ──────────────────────────────────────────────────

echo "========================================="
echo "  Tier 1: No GPU required"
echo "========================================="
echo ""

if run_test "Tier1" "C++ unit tests" "${BUILD_DIR}/hotswap_test"; then
  TIER1_PASSED=$((TIER1_PASSED + 1))
else
  TIER1_FAILED=$((TIER1_FAILED + 1))
fi

if run_test "Tier1" "Binary lifter" "${BUILD_DIR}/lifter_test"; then
  TIER1_PASSED=$((TIER1_PASSED + 1))
else
  TIER1_FAILED=$((TIER1_FAILED + 1))
fi

if run_test "Tier1" "waveasm MLIR dialect" "${BUILD_DIR}/waveasm_roundtrip_test"; then
  TIER1_PASSED=$((TIER1_PASSED + 1))
else
  TIER1_FAILED=$((TIER1_FAILED + 1))
fi

if run_test "Tier1" "Python mnemonic translation" python3 "${SCRIPT_DIR}/tests/test_transpiler.py"; then
  TIER1_PASSED=$((TIER1_PASSED + 1))
else
  TIER1_FAILED=$((TIER1_FAILED + 1))
fi

echo ""

# ── Tier 2: GPU required ─────────────────────────────────────────────────────

if [ "$NO_GPU" -eq 1 ]; then
  echo "========================================="
  echo "  Tier 2: Skipped (--no-gpu)"
  echo "========================================="
else
  echo "========================================="
  echo "  Tier 2: GPU required"
  echo "========================================="
  echo ""

  if run_test "Tier2" "Python e2e transpiler" python3 "${SCRIPT_DIR}/tests/test_transpiler_e2e.py"; then
    TIER2_PASSED=$((TIER2_PASSED + 1))
  else
    TIER2_FAILED=$((TIER2_FAILED + 1))
  fi
fi

echo ""

# ── Summary ───────────────────────────────────────────────────────────────────

echo "========================================="
echo "  Summary"
echo "========================================="
TOTAL_PASSED=$((TIER1_PASSED + TIER2_PASSED))
TOTAL_FAILED=$((TIER1_FAILED + TIER2_FAILED))
echo "  Tier 1: ${TIER1_PASSED} passed, ${TIER1_FAILED} failed"
if [ "$NO_GPU" -eq 0 ]; then
  echo "  Tier 2: ${TIER2_PASSED} passed, ${TIER2_FAILED} failed"
else
  echo "  Tier 2: skipped"
fi
echo "  Total:  ${TOTAL_PASSED} passed, ${TOTAL_FAILED} failed"
echo "========================================="

if [ "$TOTAL_FAILED" -gt 0 ]; then
  exit 1
fi
