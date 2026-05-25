#!/usr/bin/env bash
# Build, run, and disassemble coop_bitcode_test.cpp, plus produce a
# textual .ll dump of the bitcode artifact for inspection.
#
# Bitcode path: <nccl_device_wrapper.h> forward-declares the bucket B
# thunks; the bodies (and their per-Impl vtable statics + trampolines)
# live in librccl_device.bc and are linked at AMDGPU LTO time via
# -Xoffload-linker. The kernel makes indirect calls into the bitcode
# for both init and the size/thread_rank accessors.
#
# Outputs (in $OUTDIR):
#   coop_bitcode.exe                                 binary
#   coop_bitcode.amdgpu                              offload bundle (AMDGPU ELF)
#   coop_bitcode.{thread,warp,cta}.disasm            per-kernel disassembly
#   librccl_device.ll                                textual IR of the bitcode
#
# Requires librccl_device.bc to have been built (cmake -DEMIT_LLVM_IR=ON).
# Bucket B (ncclCoopAny) is unconditionally compiled into the artifact.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
BUILD="${BUILD:-$REPO/build/release}"
HIPIFIED_INC="$BUILD/hipify/src/include"
GENERATED_INC="$BUILD/include"
WRAPPER_INC="$BUILD/include"
BC="${BC:-$BUILD/lib/librccl_device.bc}"

ARCH="${ARCH:-gfx942}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
GPU="${GPU:-0}"
OUTDIR="${OUTDIR:-/tmp/coop_bitcode}"
mkdir -p "$OUTDIR"

EXE="$OUTDIR/coop_bitcode.exe"
AMDGPU="$OUTDIR/coop_bitcode.amdgpu"
LL="$OUTDIR/librccl_device.ll"

if [[ ! -f "$BC" ]]; then
  echo "ERROR: bitcode not found at $BC" >&2
  echo "Build it: cmake -DEMIT_LLVM_IR=ON -DBITCODE_LIB_ARCH=$ARCH ." >&2
  echo "        : cmake --build . --target llvm_ir" >&2
  exit 2
fi

# Cheap up-front check that the bitcode actually contains bucket B
# symbols. If we got pointed at a bucket-A-only bitcode the test would
# fail at link time with "undefined symbol"; this catches the misconfig
# earlier with a clearer message.
if ! "$ROCM_PATH/llvm/bin/llvm-nm" "$BC" 2>/dev/null \
       | grep -q -E '^[^[:space:]]+[[:space:]]+T[[:space:]]+ncclCoopAnyInitThread$'; then
  echo "ERROR: bitcode at $BC has no bucket B symbols." >&2
  echo "Rebuild librccl_device.bc with: cmake -DEMIT_LLVM_IR=ON --build . --target llvm_ir" >&2
  exit 2
fi

echo "[coop-bitcode] arch=$ARCH  GPU=$GPU  bc=$BC ($(stat -c%s "$BC") bytes)"

# Dump the bitcode as textual IR for inspection.
"$ROCM_PATH/llvm/bin/llvm-dis" "$BC" -o "$LL"
echo "[coop-bitcode] librccl_device.bc as text -> $LL"

# Build the consumer. -Xoffload-linker $BC routes the bitcode into the
# device-side link; the internalize=false plugin opt keeps the exported
# thunks alive through AMDGPU LTO.
"$ROCM_PATH/bin/hipcc" --offload-arch="$ARCH" -O2 \
  -D__HIP_PLATFORM_AMD__=1 \
  -I"$HIPIFIED_INC" \
  -I"$HIPIFIED_INC/nccl_device" \
  -I"$WRAPPER_INC" \
  -I"$GENERATED_INC" \
  "$HERE/coop_bitcode_test.cpp" \
  -Xoffload-linker "$BC" \
  -Xoffload-linker -plugin-opt=-amdgpu-internalize-symbols=false \
  -o "$EXE"

echo "[coop-bitcode] built: $EXE"

echo "[coop-bitcode] running..."
HIP_VISIBLE_DEVICES="$GPU" "$EXE"

echo "[coop-bitcode] extracting offload bundle..."
"$ROCM_PATH/llvm/bin/llvm-objdump" --offloading "$EXE" >/dev/null
cp -f "$EXE.0.hipv4-amdgcn-amd-amdhsa--$ARCH" "$AMDGPU"

disasm_one() {
  local sym=$1 out=$2
  "$ROCM_PATH/llvm/bin/llvm-objdump" \
    -d --disassemble-symbols="$sym" "$AMDGPU" > "$out" 2>/dev/null \
    || "$ROCM_PATH/llvm/bin/llvm-objdump" -d "$AMDGPU" > "$out"
}
disasm_one _Z13k_coop_threadP9CoopProbe "$OUTDIR/coop_bitcode.thread.disasm"
disasm_one _Z11k_coop_warpP9CoopProbe   "$OUTDIR/coop_bitcode.warp.disasm"
disasm_one _Z10k_coop_ctaP9CoopProbe    "$OUTDIR/coop_bitcode.cta.disasm"

echo "[coop-bitcode] disasms in $OUTDIR/coop_bitcode.{thread,warp,cta}.disasm"
echo "[coop-bitcode] PASS"
