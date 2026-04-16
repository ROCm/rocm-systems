#!/bin/bash
set -e
cd "$(dirname "$0")/build3"
LOG="/home/mluecke/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/hotswap/llvm_ir_proto/test_output.log"

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
  
  echo "=== vecadd_kernel.ll (first 80 lines) ==="
  head -n 80 /tmp/gfx1250_debug_vecadd_kernel.ll 2>&1 || echo "(missing)"
  
  echo "=== vecadd_kernel.s (first 80 lines) ==="
  head -n 80 /tmp/gfx1250_debug_vecadd_kernel.s 2>&1 || echo "(missing)"
  
  echo "=== softmax_kernel.ll (first 40 lines) ==="
  head -n 40 /tmp/gfx1250_debug_softmax_kernel.ll 2>&1 || echo "(missing)"
  
  echo "=== done ==="
} > "$LOG" 2>&1

echo "Log written to $LOG"
