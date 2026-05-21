#!/usr/bin/env bash
# Build, run, and disassemble combined_bitcode_test.cpp, plus dump the
# bitcode artifact as readable IR.
#
# Bitcode path: both bucket A (ncclGetPeerPointerTeam) and bucket B
# (ncclCoopAnyInitWarp / ncclCoopThreadRank) are extern "C" __device__
# thunks defined in librccl_device.bc and linked at AMDGPU LTO time
# via -Xoffload-linker.
#
# Outputs (in $OUTDIR):
#   combined_bitcode.exe                binary
#   combined_bitcode.amdgpu             offload bundle (AMDGPU ELF)
#   combined_bitcode.k.disasm           k_combined disassembly
#   librccl_device.ll                   textual IR of the bitcode
#
# Requires librccl_device.bc to have been built with
#     -DRCCL_ENABLE_NCCL_COOP_ANY=ON
# so it contains both bucket A and bucket B thunks.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
BUILD="${BUILD:-$REPO/build/release}"
HIPIFIED_INC="$BUILD/hipify/src/include"
GENERATED_INC="$BUILD/include"
WRAPPER_INC="$BUILD/include"
BC="${BC:-$BUILD/lib/librccl_device.bc}"

ARCH="${ARCH:-gfx942}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
GPU="${GPU:-0}"
OUTDIR="${OUTDIR:-/tmp/combined_bitcode}"
mkdir -p "$OUTDIR"

EXE="$OUTDIR/combined_bitcode.exe"
AMDGPU="$OUTDIR/combined_bitcode.amdgpu"
DISM="$OUTDIR/combined_bitcode.k.disasm"
LL="$OUTDIR/librccl_device.ll"

if [[ ! -f "$BC" ]]; then
  echo "ERROR: bitcode not found at $BC" >&2
  echo "Build it: cmake -DEMIT_LLVM_IR=ON -DBITCODE_LIB_ARCH=$ARCH \\" >&2
  echo "                -DRCCL_ENABLE_NCCL_COOP_ANY=ON ." >&2
  echo "        : cmake --build . --target llvm_ir" >&2
  exit 2
fi

# This test uses BOTH bucket A and bucket B. Spot-check both ends of
# that contract are actually in the bitcode before we build/link.
if ! "$ROCM_PATH/llvm/bin/llvm-nm" "$BC" 2>/dev/null \
       | grep -q -E '^[^[:space:]]+[[:space:]]+T[[:space:]]+ncclGetPeerPointerTeam$'; then
  echo "ERROR: bitcode missing bucket A symbol ncclGetPeerPointerTeam." >&2
  exit 2
fi
if ! "$ROCM_PATH/llvm/bin/llvm-nm" "$BC" 2>/dev/null \
       | grep -q -E '^[^[:space:]]+[[:space:]]+T[[:space:]]+ncclCoopAnyInitWarp$'; then
  echo "ERROR: bitcode missing bucket B symbol ncclCoopAnyInitWarp." >&2
  echo "Rebuild librccl_device.bc with -DRCCL_ENABLE_NCCL_COOP_ANY=ON." >&2
  exit 2
fi

echo "[combined-bitcode] arch=$ARCH  GPU=$GPU  bc=$BC ($(stat -c%s "$BC") bytes)"

"$ROCM_PATH/llvm/bin/llvm-dis" "$BC" -o "$LL"
echo "[combined-bitcode] librccl_device.bc as text -> $LL"

"$ROCM_PATH/bin/hipcc" --offload-arch="$ARCH" -O2 \
  -D__HIP_PLATFORM_AMD__=1 \
  -DRCCL_ENABLE_NCCL_COOP_ANY=1 \
  -I"$HIPIFIED_INC" \
  -I"$HIPIFIED_INC/nccl_device" \
  -I"$WRAPPER_INC" \
  -I"$GENERATED_INC" \
  "$HERE/combined_bitcode_test.cpp" \
  -Xoffload-linker "$BC" \
  -Xoffload-linker -plugin-opt=-amdgpu-internalize-symbols=false \
  -o "$EXE"

echo "[combined-bitcode] built: $EXE"

echo "[combined-bitcode] running..."
HIP_VISIBLE_DEVICES="$GPU" "$EXE"

echo "[combined-bitcode] extracting offload bundle..."
"$ROCM_PATH/llvm/bin/llvm-objdump" --offloading "$EXE" >/dev/null
cp -f "$EXE.0.hipv4-amdgcn-amd-amdhsa--$ARCH" "$AMDGPU"

echo "[combined-bitcode] disassembling kernel..."
"$ROCM_PATH/llvm/bin/llvm-objdump" \
  -d --disassemble-symbols=_Z10k_combinedPcjmiiiPPv \
  "$AMDGPU" > "$DISM" 2>/dev/null \
  || "$ROCM_PATH/llvm/bin/llvm-objdump" -d "$AMDGPU" > "$DISM"

echo "[combined-bitcode] disasm: $DISM"
echo "[combined-bitcode] PASS"
