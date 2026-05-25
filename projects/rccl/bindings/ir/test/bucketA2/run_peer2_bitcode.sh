#!/usr/bin/env bash
# Build + run peer2_bitcode_test.cpp (2-GPU bitcode bucket-A smoke).
# ncclGetPeerPointerTeam is forward-declared by <nccl_device_wrapper.h>
# and its body comes from librccl_device.bc via -Xoffload-linker.
#
# Env knobs:
#   ARCH       offload arch          (default gfx942)
#   ROCM_PATH  ROCm root             (default /opt/rocm)
#   GPUS       HIP_VISIBLE_DEVICES   (default 0,1) — must expose >=2 GPUs
#   OUTDIR     output dir            (default /tmp/peer2_bitcode)
#   BC         path to librccl_device.bc (default $BUILD/lib/librccl_device.bc)
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
GPUS="${GPUS:-0,1}"
OUTDIR="${OUTDIR:-/tmp/peer2_bitcode}"
mkdir -p "$OUTDIR"

EXE="$OUTDIR/peer2_bitcode.exe"

if [[ ! -f "$BC" ]]; then
  echo "ERROR: bitcode not found at $BC" >&2
  echo "Build with: cmake -DEMIT_LLVM_IR=ON -DBITCODE_LIB_ARCH=$ARCH ." >&2
  echo "         + cmake --build . --target llvm_ir" >&2
  exit 2
fi

echo "[peer2-bitcode] arch=$ARCH  GPUS=$GPUS  bc=$BC ($(stat -c%s "$BC") bytes)"

"$ROCM_PATH/bin/hipcc" --offload-arch="$ARCH" -O2 \
  -D__HIP_PLATFORM_AMD__=1 \
  -I"$HIPIFIED_INC" \
  -I"$HIPIFIED_INC/nccl_device" \
  -I"$WRAPPER_INC" \
  -I"$GENERATED_INC" \
  "$HERE/peer2_bitcode_test.cpp" \
  -Xoffload-linker "$BC" \
  -Xoffload-linker -plugin-opt=-amdgpu-internalize-symbols=false \
  -o "$EXE"

echo "[peer2-bitcode] built: $EXE"
echo "[peer2-bitcode] running..."
HIP_VISIBLE_DEVICES="$GPUS" "$EXE"
echo "[peer2-bitcode] done"
