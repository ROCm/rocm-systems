#!/bin/bash
# Build minimal LLVM with AMDGPU backend for AegisBit
# No Docker, no sudo required - installs to local directory

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LLVM_SRC="$PROJECT_ROOT/../llvm-project"
LLVM_BUILD="$LLVM_SRC/build"

echo "=== Building LLVM for AegisBit ==="
echo "Source: $LLVM_SRC"
echo "Build:  $LLVM_BUILD"
echo ""

if [ ! -d "$LLVM_SRC/llvm" ]; then
  echo "Error: LLVM source not found at $LLVM_SRC/llvm"
  exit 1
fi

mkdir -p "$LLVM_BUILD"
cd "$LLVM_BUILD"

# Minimal build: only AMDGPU target, only libraries we need
# No clang, no tools we don't use - keeps it fast
echo "Configuring LLVM (minimal AMDGPU-only build)..."
cmake ../llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="" \
  -DLLVM_TARGETS_TO_BUILD="AMDGPU" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_BUILD_TOOLS=OFF \
  -DLLVM_BUILD_UTILS=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DLLVM_INCLUDE_DOCS=OFF \
  -DLLVM_ENABLE_BINDINGS=OFF \
  -DLLVM_ENABLE_OCAMLDOC=OFF \
  -DLLVM_ENABLE_Z3_SOLVER=OFF \
  -G Ninja

echo ""
echo "Building LLVM..."
# Use ~25% of cores, nice priority to yield to other users
JOBS=${LLVM_BUILD_JOBS:-64}
echo "Using $JOBS parallel jobs (override with LLVM_BUILD_JOBS)"

nice -n 5 ninja -j$JOBS

echo ""
echo "=== LLVM build complete! ==="
echo "Build directory: $LLVM_BUILD"
echo ""
echo "Now build AegisBit:"
echo "  cd $PROJECT_ROOT/build"
echo "  cmake .. && make -j8"
