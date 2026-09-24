// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Manual feasibility probe, not a ConSan qualification test.
// Plain and wave-atomic access kernels demonstrate lost source-level semantics.
// The executable demonstrates why legal atomics must not receive a value-change
// race check: a positive mismatch count is the EXPECTED successful result.
// See docs/consan/SUPERCOLLIDER_GLOBAL_MEMORY.md.

#include <hip/hip_runtime.h>
extern "C" __global__ void plain_load(int *p, int *out) { out[threadIdx.x] = p[threadIdx.x]; }
extern "C" __global__ void relaxed_load(int *p, int *out) {
  out[threadIdx.x] = __atomic_load_n(p + threadIdx.x, __ATOMIC_RELAXED);
}
extern "C" __global__ void acquire_load(int *p, int *out) {
  out[threadIdx.x] = __atomic_load_n(p + threadIdx.x, __ATOMIC_ACQUIRE);
}
extern "C" __global__ void volatile_load(volatile int *p, int *out) {
  out[threadIdx.x] = p[threadIdx.x];
}
extern "C" __global__ void plain_store(int *p) { p[threadIdx.x] = 42; }
extern "C" __global__ void relaxed_store(int *p) {
  __atomic_store_n(p + threadIdx.x, 42, __ATOMIC_RELAXED);
}
extern "C" __global__ void release_store(int *p) {
  __atomic_store_n(p + threadIdx.x, 42, __ATOMIC_RELEASE);
}
extern "C" __global__ void wave_load(int *p, int *out) {
  out[threadIdx.x] =
      __hip_atomic_load(p + threadIdx.x, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
}
extern "C" __global__ void agent_load(int *p, int *out) {
  out[threadIdx.x] = __hip_atomic_load(p + threadIdx.x, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
extern "C" __global__ void plain_collision_store(int *p) { *p = threadIdx.x; }
extern "C" __global__ void wave_collision_store(int *p) {
  __hip_atomic_store(p, static_cast<int>(threadIdx.x), __ATOMIC_RELAXED,
                     __HIP_MEMORY_SCOPE_WAVEFRONT);
}

#include <cstdio>
#define HIP(x)                                                                                     \
  do {                                                                                             \
    auto e = (x);                                                                                  \
    if (e != hipSuccess) {                                                                         \
      std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e));                                  \
      return 1;                                                                                    \
    }                                                                                              \
  } while (0)
__global__ void legal_atomic_stores(int *p, int *mismatch) {
  const int lane = threadIdx.x;
  __hip_atomic_store(p, lane, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  const int observed = __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  mismatch[lane] = observed != lane;
}
int main() {
  hipDeviceProp_t prop{};
  HIP(hipGetDeviceProperties(&prop, 0));
  int *p, *m;
  HIP(hipMalloc(&p, sizeof(int)));
  HIP(hipMalloc(&m, 64 * sizeof(int)));
  legal_atomic_stores<<<1, prop.warpSize>>>(p, m);
  HIP(hipGetLastError());
  HIP(hipDeviceSynchronize());
  int marks[64] = {}, result;
  HIP(hipMemcpy(marks, m, prop.warpSize * sizeof(int), hipMemcpyDeviceToHost));
  HIP(hipMemcpy(&result, p, sizeof(int), hipMemcpyDeviceToHost));
  int count = 0;
  for (int i = 0; i < prop.warpSize; i++)
    count += marks[i];
  HIP(hipFree(m));
  HIP(hipFree(p));
  std::printf("%s legal atomic-store oracle=%s naive-replay-mismatches=%d/%d\n", prop.gcnArchName,
              result >= 0 && result < prop.warpSize ? "PASS" : "FAIL", count, prop.warpSize);
  return result >= 0 && result < prop.warpSize && count > 0 ? 0 : 1;
}
