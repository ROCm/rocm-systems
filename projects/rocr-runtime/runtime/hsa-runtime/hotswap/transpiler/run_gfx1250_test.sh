#!/bin/bash
set -e
cd "$(dirname "$0")/build"

{
  echo "=== cmake ==="
  cmake .. -G Ninja \
    -DCMAKE_PREFIX_PATH="/opt/rocm-7.2.1;$HOME/shared-llvm" \
    -Dhip_DIR=/opt/rocm-7.2.1/lib/cmake/hip \
    -DCMAKE_CXX_COMPILER=g++ 2>&1

  echo "=== ninja ==="
  ninja gfx1250_gpu_test 2>&1
  echo "ninja exit: $?"

  echo "=== gfx1250_gpu_test ==="
  ./gfx1250_gpu_test 2>&1 || true
  echo "test exit: $?"

  echo "=== done ==="
}

echo "Done."
