#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
rm -rf build
LLVM_DIR="${LLVM_DIR:-/work1/lmeadows/llvm}"
cmake -B build -S . -DLLVM_DIR="$LLVM_DIR"
cmake --build build
echo "Built: $(pwd)/device_linker"
