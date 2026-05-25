#!/usr/bin/env bash
# Build, run, and disassemble peer_inline_test.cpp.
#
# Inline path: <nccl_device.h> provides ncclGetPeerPointer as
# NCCL_DEVICE_INLINE so its body is inlined into our kernel by the
# consumer's clang frontend. No bitcode is involved.
#
# Outputs (in $OUTDIR):
#   peer_inline.exe                   binary
#   peer_inline.expected              host-computed expected pointers (binary)
#   peer_inline.amdgpu                offload bundle (AMDGPU ELF)
#   peer_inline.k.disasm              k_peer_inline disassembly
#
# Env knobs:
#   ARCH      offload arch (default gfx942)
#   ROCM_PATH ROCm root      (default /opt/rocm)
#   GPU       HIP_VISIBLE_DEVICES (default 0)
#   OUTDIR    output dir     (default /tmp/peer_inline)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
BUILD="${BUILD:-$REPO/build/release}"
HIPIFIED_INC="$BUILD/hipify/src/include"
GENERATED_INC="$BUILD/include"

ARCH="${ARCH:-gfx942}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
GPU="${GPU:-0}"
OUTDIR="${OUTDIR:-/tmp/peer_inline}"
mkdir -p "$OUTDIR"

EXE="$OUTDIR/peer_inline.exe"
EXP="$OUTDIR/peer_inline.expected"
AMDGPU="$OUTDIR/peer_inline.amdgpu"
DISM="$OUTDIR/peer_inline.k.disasm"

echo "[inline] arch=$ARCH  GPU=$GPU"
echo "[inline] include  -I$HIPIFIED_INC  -I$GENERATED_INC"

# Build. The hipified copy of src/include/* is what RCCL itself compiles
# against, so we point at it FIRST for byte-for-byte parity with the
# bitcode build's view of the headers; the generated dir then supplies
# nccl.h. -D__HIP_PLATFORM_AMD__=1 (with a value) matches what the
# bitcode build passes — without the explicit value, utility.h's
# `#if __HIP_PLATFORM_AMD__` treats the empty macro as a parse error.
"$ROCM_PATH/bin/hipcc" --offload-arch="$ARCH" -O2 \
  -D__HIP_PLATFORM_AMD__=1 \
  -I"$HIPIFIED_INC" \
  -I"$HIPIFIED_INC/nccl_device" \
  -I"$GENERATED_INC" \
  "$HERE/peer_inline_test.cpp" \
  -o "$EXE"

echo "[inline] built: $EXE"

# Run. PEER_DUMP makes main() write the host-expected pointers to disk
# so peer_bitcode can be cross-checked against the same oracle.
echo "[inline] running..."
HIP_VISIBLE_DEVICES="$GPU" PEER_DUMP="$EXP" "$EXE"
echo "[inline] expected pointers -> $EXP"

# Pull the AMDGPU device half out of the fat binary, then disassemble
# only the kernel. We strip the file path tail of the temp object names
# so successive builds compare cleanly.
echo "[inline] extracting offload bundle..."
"$ROCM_PATH/llvm/bin/llvm-objdump" --offloading "$EXE" >/dev/null
# llvm-objdump writes the bundle to <exe>.0.hipv4-amdgcn-amd-amdhsa--<arch>
SRC_BUNDLE="$EXE.0.hipv4-amdgcn-amd-amdhsa--$ARCH"
cp -f "$SRC_BUNDLE" "$AMDGPU"

echo "[inline] disassembling kernel..."
"$ROCM_PATH/llvm/bin/llvm-objdump" \
  -d --disassemble-symbols=_Z13k_peer_inlinePcPK8TestCaseiPPv \
  "$AMDGPU" > "$DISM" || {
    # Fall back to a name-pattern disassembly (mangling may vary by
    # toolchain) — dump everything and let the consumer grep.
    "$ROCM_PATH/llvm/bin/llvm-objdump" -d "$AMDGPU" > "$DISM"
}

echo "[inline] disasm: $DISM"
echo "[inline] PASS"
