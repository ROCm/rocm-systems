// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "hip_test_support.hpp"

#include <hip/hip_runtime.h>

// A paired safe/unsafe ISA sequence used to validate wait-counter hazard
// diagnostics. Keep the kernel names stable because report tests use them for
// dispatch attribution.
__global__ void vgpr_no_race_kernel(const float *src, float *dst);

__global__ void vgpr_race_kernel(const float *src, float *dst);

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
