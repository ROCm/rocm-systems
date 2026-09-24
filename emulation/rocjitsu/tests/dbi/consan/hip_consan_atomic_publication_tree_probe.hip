// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Manual gfx1201 publication journal probe: three non-returning producers.
#include <cstdio>
#include <hip/hip_runtime.h>

__global__ void publication(int *counter, int *output) {
  __shared__ volatile int partials[4];
  const int wave = threadIdx.x / 32;
  if (threadIdx.x % 32 != 0)
    return;
  partials[wave] = 10 + wave;
  if (wave < 3) {
    // Keep the no-return producer form explicit; the journal must obtain its
    // old value from this very RMW, without executing a second operation.
    const int bit = 1 << wave;
    asm volatile("s_wait_storecnt_dscnt 0\n\t"
                 "global_atomic_or_b32 %0, %1, off scope:SCOPE_DEV\n\t"
                 :
                 : "v"(counter), "v"(bit)
                 : "memory");
  } else {
    while (__hip_atomic_load(counter, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != 7) {
    }
    const int old = __hip_atomic_fetch_or(counter, 8, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    if (old == 7)
      *output = partials[0] + partials[1] + partials[2] + partials[3];
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
  int host = 0;
  HIP(hipMalloc(&counter, sizeof(int)));
  HIP(hipMalloc(&output, sizeof(int)));
  HIP(hipMemset(counter, 0, sizeof(int)));
  HIP(hipMemset(output, 0, sizeof(int)));
  publication<<<1, 128>>>(counter, output);
  HIP(hipDeviceSynchronize());
  HIP(hipMemcpy(&host, output, sizeof(int), hipMemcpyDeviceToHost));
  HIP(hipFree(counter));
  HIP(hipFree(output));
  if (host != 46)
    return 3;
  std::puts("NUMERICAL PASS");
}
