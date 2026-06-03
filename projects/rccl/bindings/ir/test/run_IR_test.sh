#!/usr/bin/env bash
# Run the comprehensive librccl_device.bc API test (IR_test.cpp).
#
# Exercises every active API from nccl_device_wrapper__impl.h and
# reports PASS / FAIL / SKIP per function.
#
# Prerequisites:
#   - librccl_device.bc has been built (cmake -DEMIT_LLVM_IR=ON).
#   - The RCCL CMake build was run at least once so that the hipify-staged
#     nccl_device headers and the generated rccl.h exist under $BUILD.
#   - ROCm is installed under $ROCM_PATH (default /opt/rocm).
#
# Env knobs:
#   ARCH       offload arch / bitcode target  (default gfx950)
#   ROCM_PATH  ROCm install root              (default /opt/rocm)
#   BUILD      RCCL CMake build directory     (default <repo>/build/release)
#   BC         path to librccl_device.bc      (default $BUILD/lib/librccl_device.bc)
#   GPU        HIP_VISIBLE_DEVICES value      (default: unset — all GPUs)
#   OUTDIR     directory for the compiled exe (default /tmp/ir_test)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"

ARCH="${ARCH:-gfx950}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
BUILD="${BUILD:-$REPO/build/release}"
BC="${BC:-$BUILD/lib/librccl_device.bc}"
OUTDIR="${OUTDIR:-/tmp/ir_test}"
mkdir -p "$OUTDIR"

EXE="$OUTDIR/IR_test.exe"
LL="$OUTDIR/librccl_device.ll"

# ── Preflight checks ──────────────────────────────────────────────────

if [[ ! -f "$BC" ]]; then
  echo "ERROR: bitcode not found at $BC" >&2
  echo "Build it: cmake -DEMIT_LLVM_IR=ON -DBITCODE_LIB_ARCH=$ARCH ." >&2
  echo "        : cmake --build . --target llvm_ir" >&2
  exit 2
fi

HIPIFIED_INC="$BUILD/hipify/src/include"
GENERATED_INC="$BUILD/include"

if [[ ! -d "$HIPIFIED_INC" ]]; then
  echo "ERROR: hipify staging directory not found: $HIPIFIED_INC" >&2
  echo "Run the RCCL CMake build at least once to populate it." >&2
  exit 2
fi

if [[ ! -f "$GENERATED_INC/rccl/rccl.h" && ! -f "$GENERATED_INC/nccl.h" ]]; then
  echo "ERROR: generated rccl.h / nccl.h not found under $GENERATED_INC" >&2
  echo "Run the RCCL CMake build at least once to generate it." >&2
  exit 2
fi

echo "[IR_test] arch        = $ARCH"
echo "[IR_test] bc          = $BC  ($(stat -c%s "$BC") bytes)"
echo "[IR_test] hipified    = $HIPIFIED_INC"
echo "[IR_test] generated   = $GENERATED_INC"
echo "[IR_test] exe         = $EXE"
echo ""

# ── Optional: dump bitcode symbols for a quick sanity check ───────────

LLVM_NM="${ROCM_PATH}/llvm/bin/llvm-nm"
if [[ -x "$LLVM_NM" ]]; then
  echo "[IR_test] Public symbols in bitcode:"
  "$LLVM_NM" "$BC" 2>/dev/null \
    | awk '/[[:space:]]T[[:space:]]nccl/ { printf "             %s\n", $3 }' \
    || true
  echo ""
fi

# ── Disassemble bitcode to .ll for offline inspection ─────────────────

"$ROCM_PATH/llvm/bin/llvm-dis" "$BC" -o "$LL" 2>/dev/null \
  && echo "[IR_test] IR dump -> $LL" || true

# ── Compile the test ──────────────────────────────────────────────────
#
# Include-path order:
#   1. hipify/src/include          — hipified nccl_device/*.h
#   2. hipify/src/include/nccl_device — impl/ sub-headers
#   3. build/include               — nccl_device_wrapper.h + generated nccl.h
#
# -Xoffload-linker $BC
#   Routes the bitcode to the AMDGPU device-side LTO link.
# -plugin-opt=-amdgpu-internalize-symbols=false
#   Prevents AMDGPU LTO from re-internalizing our exported thunks after
#   they have already been linked in.

echo "[IR_test] Compiling IR_test.cpp..."
# NOTE: Compiling at -O0 instead of -O2/-O1.
# ROCm 7.x AMDGPU backend has a codegen bug that manifests at -O2 / -O1 when
# kernels make indirect function calls (vtable dispatch) and then store results:
#   - At -O2: missing s_waitcnt lgkmcnt(0) before the store → stale SGPR base
#     pointer (gfx942 XNACK retries forever → hang), or incorrect s[54:55]
#     clobbering in the vtable dispatch loop → illegal memory access (error 700).
#   - At -O1: similar register-tracking issues cause hangs.
#   - At -O0: no optimization; the compiler emits correct register tracking.
# This is a functional-correctness test, not a benchmark, so -O0 is fine.
"$ROCM_PATH/bin/hipcc" \
  --offload-arch="$ARCH" -O0 \
  -D__HIP_PLATFORM_AMD__=1 \
  -I"$HIPIFIED_INC" \
  -I"$HIPIFIED_INC/nccl_device" \
  -I"$GENERATED_INC" \
  "$HERE/IR_test.cpp" \
  -Xoffload-linker "$BC" \
  -Xoffload-linker -plugin-opt=-amdgpu-internalize-symbols=false \
  -o "$EXE"

echo "[IR_test] Compiled: $EXE"
echo ""

# ── Run the test ──────────────────────────────────────────────────────

if [[ -n "${GPU:-}" ]]; then
  echo "[IR_test] HIP_VISIBLE_DEVICES=$GPU"
  HIP_VISIBLE_DEVICES="$GPU" "$EXE"
else
  "$EXE"
fi

echo ""
echo "[IR_test] Done.  Exit code: $?"
