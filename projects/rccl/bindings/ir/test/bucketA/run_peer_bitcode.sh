#!/usr/bin/env bash
# Build, run, and disassemble peer_bitcode_test.cpp, plus produce a
# textual .ll dump of the bitcode artifact for inspection.
#
# Bitcode path: <nccl_device_wrapper.h> only forward-declares the thunks;
# the bodies live in librccl_device.bc and are linked at AMDGPU LTO time
# via -Xoffload-linker. This is the same recipe the smoke test uses.
#
# Outputs (in $OUTDIR):
#   peer_bitcode.exe                  binary
#   peer_bitcode.expected             host-computed expected pointers
#   peer_bitcode.amdgpu               offload bundle (AMDGPU ELF)
#   peer_bitcode.k.disasm             k_peer_bitcode disassembly
#   librccl_device.ll                 textual IR of the bitcode artifact
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
BUILD="${BUILD:-$REPO/build/release}"
HIPIFIED_INC="$BUILD/hipify/src/include"
GENERATED_INC="$BUILD/include"
WRAPPER_INC="$BUILD/include"          # nccl_device_wrapper.h is staged here
BC="${BC:-$BUILD/lib/librccl_device.bc}"

ARCH="${ARCH:-gfx942}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
GPU="${GPU:-0}"
OUTDIR="${OUTDIR:-/tmp/peer_bitcode}"
mkdir -p "$OUTDIR"

EXE="$OUTDIR/peer_bitcode.exe"
EXP="$OUTDIR/peer_bitcode.expected"
AMDGPU="$OUTDIR/peer_bitcode.amdgpu"
DISM="$OUTDIR/peer_bitcode.k.disasm"
LL="$OUTDIR/librccl_device.ll"

if [[ ! -f "$BC" ]]; then
  echo "ERROR: bitcode not found at $BC" >&2
  echo "Build it: cmake -DEMIT_LLVM_IR=ON -DBITCODE_LIB_ARCH=$ARCH ." >&2
  echo "        : cmake --build . --target llvm_ir" >&2
  exit 2
fi

echo "[bitcode] arch=$ARCH  GPU=$GPU  bc=$BC ($(stat -c%s "$BC") bytes)"

# Disassemble the bitcode to readable IR for the user. Same artifact the
# linker actually consumes; just rendered as text.
"$ROCM_PATH/llvm/bin/llvm-dis" "$BC" -o "$LL"
echo "[bitcode] librccl_device.bc as text -> $LL"

# Build the consumer.
#   -Xoffload-linker $BC : feed the bitcode to the AMDGPU device-side
#                          linker (lld) so its symbols are in scope for
#                          the device-side LTO link.
#   -plugin-opt=-amdgpu-internalize-symbols=false :
#                          AMDGPU LTO normally re-internalizes everything
#                          unreferenced from the kernel; turn it off so
#                          our exported thunks survive even when fully
#                          inlined away in the kernel.
"$ROCM_PATH/bin/hipcc" --offload-arch="$ARCH" -O2 \
  -D__HIP_PLATFORM_AMD__=1 \
  -I"$HIPIFIED_INC" \
  -I"$HIPIFIED_INC/nccl_device" \
  -I"$WRAPPER_INC" \
  -I"$GENERATED_INC" \
  "$HERE/peer_bitcode_test.cpp" \
  -Xoffload-linker "$BC" \
  -Xoffload-linker -plugin-opt=-amdgpu-internalize-symbols=false \
  -o "$EXE"

echo "[bitcode] built: $EXE"

echo "[bitcode] running..."
HIP_VISIBLE_DEVICES="$GPU" PEER_DUMP="$EXP" "$EXE"
echo "[bitcode] expected pointers -> $EXP"

echo "[bitcode] extracting offload bundle..."
"$ROCM_PATH/llvm/bin/llvm-objdump" --offloading "$EXE" >/dev/null
SRC_BUNDLE="$EXE.0.hipv4-amdgcn-amd-amdhsa--$ARCH"
cp -f "$SRC_BUNDLE" "$AMDGPU"

echo "[bitcode] disassembling kernel..."
"$ROCM_PATH/llvm/bin/llvm-objdump" \
  -d --disassemble-symbols=_Z14k_peer_bitcodePcPK8TestCaseiPPv \
  "$AMDGPU" > "$DISM" || {
    "$ROCM_PATH/llvm/bin/llvm-objdump" -d "$AMDGPU" > "$DISM"
}

echo "[bitcode] disasm: $DISM"
echo "[bitcode] PASS"
