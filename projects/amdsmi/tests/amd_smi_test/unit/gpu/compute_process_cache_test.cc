// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// The compute-process cache registry keeps one cache per GPU and owns it. Under
// ASAN, LeakSanitizer fails the amdsmitst run if the registry stops freeing its
// caches.

#include <gtest/gtest.h>

#include <cstdint>

#include "amd_smi/impl/amd_smi_process_cache_testing.h"

namespace {

// Ids no real GPU uses, so no device test in the same run shares these caches.
constexpr uint32_t kGpuA = 0xFFFFFF00;
constexpr uint32_t kGpuB = 0xFFFFFF01;

TEST(GpuUnit, ComputeProcessCacheReusedPerGpu) {
  auto* first = amd::smi::get_compute_process_cache(kGpuA);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(amd::smi::get_compute_process_cache(kGpuA), first);
}

TEST(GpuUnit, ComputeProcessCacheDistinctPerGpu) {
  EXPECT_NE(amd::smi::get_compute_process_cache(kGpuA), amd::smi::get_compute_process_cache(kGpuB));
}

}  // namespace
