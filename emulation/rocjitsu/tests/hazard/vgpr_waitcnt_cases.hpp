// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "hip_test_support.hpp"

#include <hip/hip_runtime.h>

// A paired safe/unsafe ISA sequence used to validate wait-counter hazard
// diagnostics. Keep the kernel names stable because report tests use them for
// dispatch attribution.
__global__ void vgpr_no_race_kernel(const float *src, float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float val;
  asm volatile("global_load_dword %0, %1, off\n"
               "s_waitcnt vmcnt(0)\n"
               : "=v"(val)
               : "v"(&src[tid]));
  dst[tid] = val;
}

__global__ void vgpr_race_kernel(const float *src, float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float val;
  asm volatile("global_load_dword %0, %1, off\n"
               // BUG: missing s_waitcnt vmcnt(0).
               "s_nop 0\n"
               : "=v"(val)
               : "v"(&src[tid]));
  dst[tid] = val;
}

namespace rocjitsu::test {

inline void runVgprWaitcntSafe(HipHazardTestBase &test, int thread_count) {
  auto *src = test.allocWithData<float>(thread_count);
  auto *dst = test.alloc<float>(thread_count);
  vgpr_no_race_kernel<<<1, thread_count>>>(src, dst);
  test.sync();
}

inline void runVgprWaitcntHazard(HipHazardTestBase &test, int thread_count) {
  auto *src = test.allocWithData<float>(thread_count);
  auto *dst = test.alloc<float>(thread_count);
  vgpr_race_kernel<<<1, thread_count>>>(src, dst);
  test.sync();
}

} // namespace rocjitsu::test
