/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * ROCM-26107: hipGetMipmappedArrayLevel memory leak regression tests.
 *
 * hipGetMipmappedArrayLevel creates a view that retains the parent image.
 * hipFreeMipmappedArray must release these views so the parent refcount
 * reaches zero and VRAM is freed. These tests verify that no VRAM leaks
 * when level views are obtained and the mipmapped array is freed.
 */

#include <hip_test_common.hh>
#include "hipArrayCommon.hh"

static constexpr int kLeakTestIters = 50;
static constexpr size_t kLeakTestWidth = 256;
static constexpr size_t kLeakTestHeight = 256;
static constexpr unsigned int kLeakTestLevels = 5;
// Threshold in bytes: allow small variance from allocator fragmentation
static constexpr size_t kLeakThresholdBytes = 512 * 1024;  // 512 KB

/*
 * Regression test: alloc mipmapped array, get level 0, free.
 * VRAM before and after should match (within threshold).
 * This is the minimal reproducer for ROCM-26107.
 */
HIP_TEMPLATE_TEST_CASE(Unit_hipFreeMipmappedArray_NoLeakAfterGetLevel, float, float4) {
  hipChannelFormatDesc desc = hipCreateChannelDesc<TestType>();
  hipExtent extent = make_hipExtent(kLeakTestWidth, kLeakTestHeight, 0);

  // Warm up: do one alloc+getLevel+free cycle so any one-time allocations are done
  {
    hipMipmappedArray_t warmup = nullptr;
    HIP_CHECK_IGNORED_RETURN(
        hipMallocMipmappedArray(&warmup, &desc, extent, kLeakTestLevels, hipArrayDefault),
        hipErrorNotSupported);
    hipArray_t lvl = nullptr;
    HIP_CHECK(hipGetMipmappedArrayLevel(&lvl, warmup, 0));
    HIP_CHECK(hipFreeMipmappedArray(warmup));
  }

  HIP_CHECK(hipDeviceSynchronize());
  size_t free_before, total;
  HIP_CHECK(hipMemGetInfo(&free_before, &total));

  for (int i = 0; i < kLeakTestIters; i++) {
    hipMipmappedArray_t mipmap = nullptr;
    HIP_CHECK(hipMallocMipmappedArray(&mipmap, &desc, extent, kLeakTestLevels, hipArrayDefault));

    hipArray_t level0 = nullptr;
    HIP_CHECK(hipGetMipmappedArrayLevel(&level0, mipmap, 0));

    HIP_CHECK(hipFreeMipmappedArray(mipmap));
  }

  HIP_CHECK(hipDeviceSynchronize());
  size_t free_after;
  HIP_CHECK(hipMemGetInfo(&free_after, &total));

  size_t leaked = (free_before > free_after) ? (free_before - free_after) : 0;
  INFO("Leaked bytes after " << kLeakTestIters << " iterations: " << leaked);
  REQUIRE(leaked < kLeakThresholdBytes);
}

/*
 * Regression test: alloc mipmapped array, get ALL levels, free.
 * Verifies that multiple level views don't accumulate leaked VRAM.
 */
HIP_TEST_CASE(Unit_hipFreeMipmappedArray_NoLeakAfterGetAllLevels) {
  hipChannelFormatDesc desc = hipCreateChannelDesc<float4>();
  hipExtent extent = make_hipExtent(kLeakTestWidth, kLeakTestHeight, 0);

  // Warm up
  {
    hipMipmappedArray_t warmup = nullptr;
    HIP_CHECK_IGNORED_RETURN(
        hipMallocMipmappedArray(&warmup, &desc, extent, kLeakTestLevels, hipArrayDefault),
        hipErrorNotSupported);
    for (unsigned int lvl = 0; lvl < kLeakTestLevels; lvl++) {
      hipArray_t arr = nullptr;
      HIP_CHECK(hipGetMipmappedArrayLevel(&arr, warmup, lvl));
    }
    HIP_CHECK(hipFreeMipmappedArray(warmup));
  }

  HIP_CHECK(hipDeviceSynchronize());
  size_t free_before, total;
  HIP_CHECK(hipMemGetInfo(&free_before, &total));

  for (int i = 0; i < kLeakTestIters; i++) {
    hipMipmappedArray_t mipmap = nullptr;
    HIP_CHECK(hipMallocMipmappedArray(&mipmap, &desc, extent, kLeakTestLevels, hipArrayDefault));

    for (unsigned int lvl = 0; lvl < kLeakTestLevels; lvl++) {
      hipArray_t levelArr = nullptr;
      HIP_CHECK(hipGetMipmappedArrayLevel(&levelArr, mipmap, lvl));
    }

    HIP_CHECK(hipFreeMipmappedArray(mipmap));
  }

  HIP_CHECK(hipDeviceSynchronize());
  size_t free_after;
  HIP_CHECK(hipMemGetInfo(&free_after, &total));

  size_t leaked = (free_before > free_after) ? (free_before - free_after) : 0;
  INFO("Leaked bytes after " << kLeakTestIters << " iterations (all " << kLeakTestLevels
                             << " levels): " << leaked);
  REQUIRE(leaked < kLeakThresholdBytes);
}

/*
 * Regression test: call hipGetMipmappedArrayLevel twice for the same level.
 * Verifies that duplicate view creation doesn't cause double-leak.
 */
HIP_TEST_CASE(Unit_hipFreeMipmappedArray_NoLeakAfterDuplicateGetLevel) {
  hipChannelFormatDesc desc = hipCreateChannelDesc<float4>();
  hipExtent extent = make_hipExtent(kLeakTestWidth, kLeakTestHeight, 0);

  // Warm up
  {
    hipMipmappedArray_t warmup = nullptr;
    HIP_CHECK_IGNORED_RETURN(
        hipMallocMipmappedArray(&warmup, &desc, extent, kLeakTestLevels, hipArrayDefault),
        hipErrorNotSupported);
    hipArray_t a = nullptr, b = nullptr;
    HIP_CHECK(hipGetMipmappedArrayLevel(&a, warmup, 0));
    HIP_CHECK(hipGetMipmappedArrayLevel(&b, warmup, 0));
    HIP_CHECK(hipFreeMipmappedArray(warmup));
  }

  HIP_CHECK(hipDeviceSynchronize());
  size_t free_before, total;
  HIP_CHECK(hipMemGetInfo(&free_before, &total));

  for (int i = 0; i < kLeakTestIters; i++) {
    hipMipmappedArray_t mipmap = nullptr;
    HIP_CHECK(hipMallocMipmappedArray(&mipmap, &desc, extent, kLeakTestLevels, hipArrayDefault));

    hipArray_t level0_a = nullptr;
    hipArray_t level0_b = nullptr;
    HIP_CHECK(hipGetMipmappedArrayLevel(&level0_a, mipmap, 0));
    HIP_CHECK(hipGetMipmappedArrayLevel(&level0_b, mipmap, 0));

    HIP_CHECK(hipFreeMipmappedArray(mipmap));
  }

  HIP_CHECK(hipDeviceSynchronize());
  size_t free_after;
  HIP_CHECK(hipMemGetInfo(&free_after, &total));

  size_t leaked = (free_before > free_after) ? (free_before - free_after) : 0;
  INFO("Leaked bytes after " << kLeakTestIters << " iterations (duplicate getLevel): " << leaked);
  REQUIRE(leaked < kLeakThresholdBytes);
}
