#!/bin/bash
set -euo pipefail

WT=~/airuntime28-clr
BUILD=~/airuntime28-clr/build
INSTALL=~/airuntime28-clr-install

mkdir -p "$BUILD"

echo "=== configure ==="
cmake -S "$WT/projects/clr" -B "$BUILD" -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_INSTALL_PREFIX="$INSTALL" \
  -DROCM_PATH=/opt/rocm \
  -DCLR_BUILD_HIP=ON \
  -DCLR_BUILD_OCL=OFF \
  -DHIP_COMMON_DIR="$WT/projects/hip" \
  -DHIP_PLATFORM=amd \
  -DHIPCC_BIN_DIR=/opt/rocm/bin \
  -DCMAKE_HIP_ARCHITECTURES=gfx1250 \
  2>&1 | tail -25

echo
echo "=== build ==="
cmake --build "$BUILD" --parallel "$(nproc)" 2>&1 | tail -30

echo
echo "=== install ==="
cmake --install "$BUILD" 2>&1 | tail -5

echo
ls -la "$INSTALL/lib/libamdhip64.so"* 2>/dev/null || ls -la "$BUILD"/lib/libamdhip64.so* 2>/dev/null
echo "CLR_BUILD_OK"
