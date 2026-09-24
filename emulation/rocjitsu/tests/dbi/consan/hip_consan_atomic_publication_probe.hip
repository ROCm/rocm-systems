// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Manual gfx1201 diagnostic for the known atomic-publication false positive.
// See docs/consan/validation/ATOMIC_PUBLICATION_RDNA4_ANALYSIS.md.
// COUNT, BARRIER, RELAXED, EXTRA_FENCE, ACTIVE_LANES, OR_COUNTER, and
// ATOMIC_REPEATS select comparison and journal-capacity arms.
#include <cstdio>
#include <hip/hip_runtime.h>
#ifndef EXTRA_FENCE
#define EXTRA_FENCE 0
#endif
#ifndef ACTIVE_LANES
#define ACTIVE_LANES 1
#endif
#ifndef OR_COUNTER
#define OR_COUNTER 0
#endif
#ifndef ATOMIC_REPEATS
#define ATOMIC_REPEATS 1
#endif
static_assert(ATOMIC_REPEATS >= 1 && (!OR_COUNTER || ATOMIC_REPEATS == 1));
static_assert(ACTIVE_LANES >= 1 && ACTIVE_LANES <= 32);
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
  __shared__ volatile int partials[2][ACTIVE_LANES][COUNT];
  int wave = threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  int old = -1;
  if (lane < ACTIVE_LANES) {
#pragma unroll
    for (int i = 0; i < COUNT; ++i)
      partials[wave][lane][i] = 10 + wave + i;
    if (EXTRA_FENCE)
      __threadfence();
#pragma unroll 1
    for (int repeat = 0; repeat < ATOMIC_REPEATS; ++repeat) {
      if (OR_COUNTER)
        old = __hip_atomic_fetch_or(counter + lane, 1 << wave,
                                    RELAXED ? __ATOMIC_RELAXED : __ATOMIC_ACQ_REL,
                                    __HIP_MEMORY_SCOPE_AGENT);
      else
        old =
            __hip_atomic_fetch_add(counter + lane, 1, RELAXED ? __ATOMIC_RELAXED : __ATOMIC_ACQ_REL,
                                   __HIP_MEMORY_SCOPE_AGENT);
    }
  }
#if BARRIER
  __syncthreads();
#endif
  if (lane < ACTIVE_LANES && (OR_COUNTER ? old > 0 : old == 2 * ATOMIC_REPEATS - 1)) {
#pragma unroll
    for (int i = 0; i < COUNT; ++i)
      output[lane * COUNT + i] = partials[0][lane][i] + partials[1][lane][i];
  }
}
#define HIP(call)                                                                                  \
  do {                                                                                             \
    hipError_t err = (call);                                                                       \
    if (err != hipSuccess) {                                                                       \
      std::fprintf(stderr, "%s\n", hipGetErrorString(err));                                        \
      return 2;                                                                                    \
    }                                                                                              \
  } while (0)
int main() {
  int *counter, *output;
  int host[ACTIVE_LANES * COUNT];
  HIP(hipMalloc(&counter, ACTIVE_LANES * sizeof(int)));
  HIP(hipMalloc(&output, sizeof(host)));
  HIP(hipMemset(counter, 0, ACTIVE_LANES * sizeof(int)));
  HIP(hipMemset(output, 0, sizeof(host)));
  publication<<<1, 64>>>(counter, output);
  HIP(hipDeviceSynchronize());
  HIP(hipMemcpy(host, output, sizeof(host), hipMemcpyDeviceToHost));
  for (int lane = 0; lane < ACTIVE_LANES; ++lane)
    for (int i = 0; i < COUNT; ++i)
      if (host[lane * COUNT + i] != 21 + 2 * i)
        return 3;
  HIP(hipFree(counter));
  HIP(hipFree(output));
  std::puts("NUMERICAL PASS");
}
