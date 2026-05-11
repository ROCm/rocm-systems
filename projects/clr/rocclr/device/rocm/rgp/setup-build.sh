#!/usr/bin/env bash
# setup-build.sh — One-shot setup for the CLR RGP/UberTrace profiler build on Windows.
#
# Run from any directory. Creates/reuses C:/github-emu/hipamd as the build tree.
#
# Prerequisites:
#   - Visual Studio 2026 (MSVC 18) installed
#   - CMake on PATH
#   - C:/opt/rocm installed (provides clang, hipcc)
#   - C:/github-emu/pal checked out (PAL provides DevDriver + RDF)
#   - C:/github-emu/rdf checked out (amdrdf.h header)
#   - C:/github-emu/rocm-systems checked out on branch 'profiler'
#   - C:/github-emu/rocm-systems/shared/amdgpu-windows-interop present

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
BUILD_DIR="C:/github-emu/hipamd"
INSTALL_DIR="C:/github-emu/install"

echo "=== CLR RGP Profiler Build Setup ==="
echo "Repo:    $REPO_ROOT"
echo "Build:   $BUILD_DIR"
echo "Install: $INSTALL_DIR"
echo ""

# ── Verify prerequisites ──────────────────────────────────────────────────────
check_path() {
  if [[ ! -e "$1" ]]; then
    echo "ERROR: Required path missing: $1"
    echo "       $2"
    exit 1
  fi
}

check_path "/c/opt/rocm/bin/clang.exe"           "Install ROCm SDK to C:/opt/rocm"
check_path "/c/github-emu/pal/CMakeLists.txt"    "Checkout PAL to C:/github-emu/pal"
check_path "/c/github-emu/rdf/rdf/inc/amdrdf.h"  "Checkout RDF to C:/github-emu/rdf"
check_path "$REPO_ROOT/projects/clr/rocclr/device/rocm/rgp/rocgpuopen.cpp" \
           "Ensure you are on the 'profiler' branch with the rgp/ folder present"
check_path "$REPO_ROOT/shared/amdgpu-windows-interop" \
           "Ensure shared/amdgpu-windows-interop is present (sparse checkout or full clone)"

echo "All prerequisites found."
echo ""

# ── Create build directory ────────────────────────────────────────────────────
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# ── Configure ─────────────────────────────────────────────────────────────────
if [[ -f "CMakeCache.txt" ]]; then
  echo "CMakeCache.txt exists — skipping configure (delete it to force reconfigure)."
  echo "To reconfigure: rm $BUILD_DIR/CMakeCache.txt && bash $0"
else
  echo "Running CMake configure..."
  cmake "$REPO_ROOT/projects/clr" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCLR_BUILD_HIP=ON \
    -DHIP_COMMON_DIR="$REPO_ROOT/projects/hip" \
    -DHIPCC_BIN_DIR="c:/opt/rocm/bin" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -D__HIP_ENABLE_PCH=OFF \
    -DROCCLR_ENABLE_HSA=ON \
    -DROCCLR_ENABLE_PAL=ON \
    -D__HIP_ENABLE_RTC=ON \
    -DUSE_PROF_API=OFF \
    -DROCR_DLL_LOAD=OFF \
    "-DAMD_COMPUTE_WIN=$REPO_ROOT/../../../shared/amdgpu-windows-interop/" \
    -DLIB_SRC_BUILD=ON \
    -DROCCLR_ENABLE_GPUOPEN=ON
  echo "Configure done."
fi

echo ""

# ── Build ─────────────────────────────────────────────────────────────────────
echo "Building rocclr (target only, faster iteration)..."
cmake --build . --config Release -j "$(nproc 2>/dev/null || echo 6)" --target rocclr

echo ""
echo "=== Build successful ==="
echo "To do a full install (amdhip64 + rocclr):"
echo "  cd $BUILD_DIR && cmake --build . --config Release -j 6 --target install"
