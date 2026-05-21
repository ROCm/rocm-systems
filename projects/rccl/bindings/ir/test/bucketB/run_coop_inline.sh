#!/usr/bin/env bash
# Build, run, and disassemble coop_inline_test.cpp.
#
# Inline path: ncclCoopAny, its vtable statics, and the per-Impl trampolines
# are all defined inline in <nccl_device.h>/coop.h. The kernel sees the
# full source and the optimizer inlines everything in one frontend pass.
# No bitcode is involved.
#
# Outputs (in $OUTDIR):
#   coop_inline.exe                                 binary
#   coop_inline.amdgpu                              offload bundle (AMDGPU ELF)
#   coop_inline.{thread,warp,cta}.disasm            per-kernel disassembly
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
BUILD="${BUILD:-$REPO/build/release}"
HIPIFIED_INC="$BUILD/hipify/src/include"
GENERATED_INC="$BUILD/include"

ARCH="${ARCH:-gfx942}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
GPU="${GPU:-0}"
OUTDIR="${OUTDIR:-/tmp/coop_inline}"
mkdir -p "$OUTDIR"

EXE="$OUTDIR/coop_inline.exe"
AMDGPU="$OUTDIR/coop_inline.amdgpu"

echo "[coop-inline] arch=$ARCH  GPU=$GPU"

# -DRCCL_ENABLE_NCCL_COOP_ANY=1 : same gate the bitcode build uses; without
#   it, coop.h does not define ncclCoopAny and the test does not compile.
# -D__HIP_PLATFORM_AMD__=1      : same as in the bucket A inline runner.
"$ROCM_PATH/bin/hipcc" --offload-arch="$ARCH" -O2 \
  -D__HIP_PLATFORM_AMD__=1 \
  -DRCCL_ENABLE_NCCL_COOP_ANY=1 \
  -I"$HIPIFIED_INC" \
  -I"$HIPIFIED_INC/nccl_device" \
  -I"$GENERATED_INC" \
  "$HERE/coop_inline_test.cpp" \
  -o "$EXE"

echo "[coop-inline] built: $EXE"

echo "[coop-inline] running..."
HIP_VISIBLE_DEVICES="$GPU" "$EXE"

echo "[coop-inline] extracting offload bundle..."
"$ROCM_PATH/llvm/bin/llvm-objdump" --offloading "$EXE" >/dev/null
cp -f "$EXE.0.hipv4-amdgcn-amd-amdhsa--$ARCH" "$AMDGPU"

# Disassemble each of the three coop kernels by mangled name. If the
# mangling ever shifts, fall back to dumping everything.
disasm_one() {
  local sym=$1 out=$2
  "$ROCM_PATH/llvm/bin/llvm-objdump" \
    -d --disassemble-symbols="$sym" "$AMDGPU" > "$out" 2>/dev/null \
    || "$ROCM_PATH/llvm/bin/llvm-objdump" -d "$AMDGPU" > "$out"
}
disasm_one _Z13k_coop_threadP9CoopProbe "$OUTDIR/coop_inline.thread.disasm"
disasm_one _Z11k_coop_warpP9CoopProbe   "$OUTDIR/coop_inline.warp.disasm"
disasm_one _Z10k_coop_ctaP9CoopProbe    "$OUTDIR/coop_inline.cta.disasm"

echo "[coop-inline] disasms in $OUTDIR/coop_inline.{thread,warp,cta}.disasm"
echo "[coop-inline] PASS"
