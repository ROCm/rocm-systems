#!/usr/bin/env bash
# Build + run peer2_inline_test.cpp (2-GPU inline bucket-A smoke).
#
# Env knobs:
#   ARCH       offload arch          (default gfx942)
#   ROCM_PATH  ROCm root             (default /opt/rocm)
#   GPUS       HIP_VISIBLE_DEVICES   (default 0,1) — must expose >=2 GPUs
#   OUTDIR     output dir            (default /tmp/peer2_inline)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
BUILD="${BUILD:-$REPO/build/release}"
HIPIFIED_INC="$BUILD/hipify/src/include"
GENERATED_INC="$BUILD/include"

ARCH="${ARCH:-gfx942}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
GPUS="${GPUS:-0,1}"
OUTDIR="${OUTDIR:-/tmp/peer2_inline}"
mkdir -p "$OUTDIR"

EXE="$OUTDIR/peer2_inline.exe"

echo "[peer2-inline] arch=$ARCH  GPUS=$GPUS"

"$ROCM_PATH/bin/hipcc" --offload-arch="$ARCH" -O2 \
  -D__HIP_PLATFORM_AMD__=1 \
  -I"$HIPIFIED_INC" \
  -I"$HIPIFIED_INC/nccl_device" \
  -I"$GENERATED_INC" \
  "$HERE/peer2_inline_test.cpp" \
  -o "$EXE"

echo "[peer2-inline] built: $EXE"
echo "[peer2-inline] running..."
HIP_VISIBLE_DEVICES="$GPUS" "$EXE"
echo "[peer2-inline] done"
