#!/bin/bash
# Build the hip-tests memory suite against the patched CLR.
set -uo pipefail

WT=~/airuntime28-clr
BUILD=~/airuntime28-hiptests
INSTALL=~/airuntime28-clr-install

mkdir -p "$BUILD"

# Configure entirely against /opt/rocm: CMake's HIP language detection wants a
# full ROCm tree, which the minimal CLR install is not. The patched runtime is
# substituted at run time via LD_LIBRARY_PATH instead, which is the same
# mechanism the e2e test used and verified with AMD_LOG_LEVEL.
echo "=== configure hip-tests ==="
cmake -S "$WT/projects/hip-tests/catch" -B "$BUILD" -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DHIP_PLATFORM=amd \
  -DHIP_PATH=/opt/rocm \
  -DROCM_PATH=/opt/rocm \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DOFFLOAD_ARCH_STR="--offload-arch=gfx1250" \
  2>&1 | tail -20

echo
echo "=== available memory targets ==="
ninja -C "$BUILD" -t targets 2>/dev/null | grep -oE '^[A-Za-z_]*[Mm]emory[A-Za-z_]*' | sort -u | head -20

echo
echo "=== build memory suite ==="
cmake --build "$BUILD" --parallel "$(nproc)" --target MemoryTest 2>&1 | tail -25

echo "HIPTESTS_BUILD_DONE"
