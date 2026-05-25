#!/usr/bin/env bash
# Build, run, and disassemble combined_inline_test.cpp.
#
# Inline path: bucket A + bucket B bodies are both supplied by
# <nccl_device.h> as inline device code. The kernel is one frontend
# pass; everything inlines.
#
# Outputs (in $OUTDIR):
#   combined_inline.exe                 binary
#   combined_inline.amdgpu              offload bundle (AMDGPU ELF)
#   combined_inline.k.disasm            k_combined disassembly
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
BUILD="${BUILD:-$REPO/build/release}"
HIPIFIED_INC="$BUILD/hipify/src/include"
GENERATED_INC="$BUILD/include"

ARCH="${ARCH:-gfx942}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
GPU="${GPU:-0}"
OUTDIR="${OUTDIR:-/tmp/combined_inline}"
mkdir -p "$OUTDIR"

EXE="$OUTDIR/combined_inline.exe"
AMDGPU="$OUTDIR/combined_inline.amdgpu"
DISM="$OUTDIR/combined_inline.k.disasm"

echo "[combined-inline] arch=$ARCH  GPU=$GPU"

# ncclCoopAny is unconditionally defined in coop.h; no extra macro needed.
"$ROCM_PATH/bin/hipcc" --offload-arch="$ARCH" -O2 \
  -D__HIP_PLATFORM_AMD__=1 \
  -I"$HIPIFIED_INC" \
  -I"$HIPIFIED_INC/nccl_device" \
  -I"$GENERATED_INC" \
  "$HERE/combined_inline_test.cpp" \
  -o "$EXE"

echo "[combined-inline] built: $EXE"

echo "[combined-inline] running..."
HIP_VISIBLE_DEVICES="$GPU" "$EXE"

echo "[combined-inline] extracting offload bundle..."
"$ROCM_PATH/llvm/bin/llvm-objdump" --offloading "$EXE" >/dev/null
cp -f "$EXE.0.hipv4-amdgcn-amd-amdhsa--$ARCH" "$AMDGPU"

echo "[combined-inline] disassembling kernel..."
"$ROCM_PATH/llvm/bin/llvm-objdump" \
  -d --disassemble-symbols=_Z10k_combinedPcjmiiiPPv \
  "$AMDGPU" > "$DISM" 2>/dev/null \
  || "$ROCM_PATH/llvm/bin/llvm-objdump" -d "$AMDGPU" > "$DISM"

echo "[combined-inline] disasm: $DISM"
echo "[combined-inline] PASS"
