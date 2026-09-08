// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Multi-wave debugging demo for the rocgdb CI harness (debug_kernel.sh).
//
// Launches a single workgroup of 128 threads = two wave64 or four wave32
// wavefronts, so a breakpoint traps every wave of one workgroup at once. Each thread derives a
// deterministic value from its global id, letting the harness assert that
// rocm-dbgapi correlates each wave to the right workgroup/position and that
// per-wave scratch (private) reads return each wave's own values.

#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void multi_wave(int *data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int local = i * 7 + 3;
  if (i < n)
    data[i] = local;
}

int main() {
  const int n = 128; // one workgroup: two wave64 or four wave32 wavefronts
  int *d = nullptr;
  if (hipMalloc(&d, n * sizeof(int)) != hipSuccess)
    return 1;
  hipLaunchKernelGGL(multi_wave, dim3(1), dim3(128), 0, 0, d, n);
  hipDeviceSynchronize();
  int h[128];
  hipMemcpy(h, d, n * sizeof(int), hipMemcpyDeviceToHost);
  printf("multi_wave done: h[0]=%d h[64]=%d h[127]=%d\n", h[0], h[64], h[127]);
  hipFree(d);
  return 0;
}
