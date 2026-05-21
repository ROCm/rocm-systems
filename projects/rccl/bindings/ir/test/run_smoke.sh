#!/usr/bin/env bash
# Reproducible runner for the librccl_device.bc smoke test.
#
# Builds smoke.cpp linked against librccl_device.bc and runs it on
# an idle GPU. Exits 0 on success.
#
# Requirements:
#   - librccl_device.bc has been built (cmake -DEMIT_LLVM_IR=ON ; make llvm_ir).
#   - librccl_device.bc was built with BITCODE_LIB_ARCH matching the target GPU,
#     overridable here via $ARCH (default gfx942).
#   - ROCm is installed under $ROCM_PATH (default /opt/rocm).
#
# Optional env:
#   ARCH       offload arch / bitcode arch (default gfx942)
#   ROCM_PATH  ROCm install root (default /opt/rocm)
#   BC         absolute path to librccl_device.bc
#              (default: <repo>/build/release/lib/librccl_device.bc)
#   GPU        HIP_VISIBLE_DEVICES value to use (default 0)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

ARCH="${ARCH:-gfx942}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
BC="${BC:-$REPO/build/release/lib/librccl_device.bc}"
GPU="${GPU:-0}"
OUT="${OUT:-/tmp/rccl_ir_smoke}"

if [[ ! -f "$BC" ]]; then
  echo "ERROR: bitcode not found at $BC" >&2
  echo "Build it with: cmake -DEMIT_LLVM_IR=ON -DBITCODE_LIB_ARCH=$ARCH ." >&2
  echo "             : cmake --build . --target llvm_ir" >&2
  exit 2
fi

echo "[smoke] arch=$ARCH"
echo "[smoke] bc=$BC ($(stat -c%s "$BC") bytes)"
echo "[smoke] HIP_VISIBLE_DEVICES=$GPU"

# Probe the bitcode for bucket B (coop) symbols. If present, compile the
# coop test cases in too. Otherwise the test is bucket A only.
COOP_FLAGS=()
LLVM_NM="${LLVM_NM:-$ROCM_PATH/llvm/bin/llvm-nm}"
if [[ -x "$LLVM_NM" ]] && "$LLVM_NM" "$BC" 2>/dev/null \
     | grep -q -E '^[^[:space:]]+[[:space:]]+T[[:space:]]+ncclCoopAnyInitThread$'; then
  echo "[smoke] bucket B detected in bitcode -> enabling SMOKE_ENABLE_COOP"
  COOP_FLAGS=(-DSMOKE_ENABLE_COOP=1)
else
  echo "[smoke] bucket B not detected -> bucket A only"
fi

# -Xoffload-linker $BC      -> route bitcode to the AMDGPU LTO link.
# -plugin-opt=-amdgpu-internalize-symbols=false
#   The AMDGPU LTO pipeline runs internalize-symbols by default, which
#   would re-internalize our exported thunks after they've already been
#   linked in. Turn it off so the thunks survive to the final binary.
"$ROCM_PATH/bin/hipcc" --offload-arch="$ARCH" -O2 \
  "${COOP_FLAGS[@]}" \
  "$HERE/smoke.cpp" \
  -Xoffload-linker "$BC" \
  -Xoffload-linker -plugin-opt=-amdgpu-internalize-symbols=false \
  -o "$OUT"

echo "[smoke] built: $OUT"
echo "[smoke] running..."
HIP_VISIBLE_DEVICES="$GPU" "$OUT"
echo "[smoke] PASS"
