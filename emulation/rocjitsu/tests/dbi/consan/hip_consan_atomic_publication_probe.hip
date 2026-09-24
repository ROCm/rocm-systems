// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Manual gfx1201 diagnostic for the known atomic-publication false positive.
// See docs/consan/validation/ATOMIC_PUBLICATION_RDNA4_ANALYSIS.md.
// COUNT, BARRIER, RELAXED, and EXTRA_FENCE select the comparison arms.
#include <hip/hip_runtime.h>
#include <cstdio>
#ifndef EXTRA_FENCE
#define EXTRA_FENCE 0
#endif
#ifndef COUNT
#define COUNT 8
#endif
#ifndef BARRIER
#define BARRIER 0
#endif
#ifndef RELAXED
#define RELAXED 0
#endif
__global__ void publication(int *counter, int *output) {
  __shared__ volatile int partials[2][COUNT];
  int wave = threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  int old = -1;
  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < COUNT; ++i) partials[wave][i] = 10 + wave + i;
    if (EXTRA_FENCE) __threadfence();
    old = __hip_atomic_fetch_add(counter, 1, RELAXED ? __ATOMIC_RELAXED : __ATOMIC_ACQ_REL,
                                __HIP_MEMORY_SCOPE_AGENT);
  }
#if BARRIER
  __syncthreads();
#endif
  if (lane == 0 && old == 1) {
#pragma unroll
    for (int i = 0; i < COUNT; ++i) output[i] = partials[0][i] + partials[1][i];
  }
}
#define HIP(call) do { hipError_t err = (call); if (err != hipSuccess) { std::fprintf(stderr, "%s\n", hipGetErrorString(err)); return 2; } } while (0)
int main() {
  int *counter, *output; int host[COUNT];
  HIP(hipMalloc(&counter, sizeof(int))); HIP(hipMalloc(&output, sizeof(host)));
  HIP(hipMemset(counter, 0, sizeof(int))); HIP(hipMemset(output, 0, sizeof(host)));
  publication<<<1,64>>>(counter, output); HIP(hipDeviceSynchronize());
  HIP(hipMemcpy(host, output, sizeof(host), hipMemcpyDeviceToHost));
  for (int i = 0; i < COUNT; ++i) if (host[i] != 21 + 2*i) return 3;
  HIP(hipFree(counter)); HIP(hipFree(output)); std::puts("NUMERICAL PASS");
}
