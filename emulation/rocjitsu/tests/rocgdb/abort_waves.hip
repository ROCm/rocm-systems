// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Multi-wave abort workload for the ROCgdb CI harness (debug_kernel.sh).
// Each wave reports an abort. The debugger must be able to inspect the first
// stop and forward every remaining stop after ROCr has handled the queue error.

#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void abort_waves() { asm volatile("s_trap 2"); }

int main() {
  if (hipSetDevice(0) != hipSuccess)
    return 1;
  abort_waves<<<1, 128>>>();
  const hipError_t status = hipDeviceSynchronize();
  if (status == hipSuccess) {
    std::fprintf(stderr, "abort_waves incorrectly synchronized successfully\n");
    return 2;
  }
  std::printf("abort_waves done: hipDeviceSynchronize reported %s\n", hipGetErrorName(status));
  return 0;
}
