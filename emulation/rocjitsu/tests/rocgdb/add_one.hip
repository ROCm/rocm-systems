// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Minimal HIP workload used by the rocgdb wave-debugging demo and CI harness
// (tests/rocgdb/debug_kernel.sh). It launches a single `add_one` kernel over one
// 64-lane wavefront and prints a deterministic result, so a debugger can set a
// breakpoint on `add_one`, inspect the stopped wave, single-step, and continue
// to a known-good output ("add_one done: host[0]=1 host[63]=1").
//
// Build with device debug info for the emulated target, e.g.:
//   hipcc --offload-arch=gfx950 -g -O0 -o add_one add_one.hip
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void add_one(int *data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    data[i] += 1;
}

int main() {
  const int n = 64;
  int *d = nullptr;
  if (hipMalloc(&d, n * sizeof(int)) != hipSuccess) {
    std::fprintf(stderr, "hipMalloc failed\n");
    return 1;
  }
  if (hipMemset(d, 0, n * sizeof(int)) != hipSuccess)
    return 2;
  add_one<<<dim3(1), dim3(64)>>>(d, n);
  if (hipDeviceSynchronize() != hipSuccess)
    return 3;
  int host[n];
  if (hipMemcpy(host, d, n * sizeof(int), hipMemcpyDeviceToHost) != hipSuccess)
    return 4;
  hipFree(d);
  std::printf("add_one done: host[0]=%d host[63]=%d\n", host[0], host[n - 1]);
  return 0;
}
