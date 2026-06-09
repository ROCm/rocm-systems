/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/* Test Case Description:
   Stress test for memory pool allocations. Allocates blocks from 1% to 50%
   of total device memory in 2% increments, verifies pool attributes
   (UsedMemCurrent, ReservedMemCurrent) after all allocations, then frees
   all blocks and verifies memory returns to zero.
*/

#include <hip_test_common.hh>

HIP_TEST_CASE(Unit_hipMemPoolMaxAlloc) {
  int device = 0;
  HIP_CHECK(hipSetDevice(device));

  hipMemPool_t pool;
  HIP_CHECK(hipDeviceGetDefaultMemPool(&pool, device));

  uint64_t threshold = 0;
  HIP_CHECK(hipMemPoolSetAttribute(pool, hipMemPoolAttrReleaseThreshold, &threshold));

  std::size_t free{}, total{};
  HIP_CHECK(hipMemGetInfo(&free, &total));
  const std::size_t memBudget = total;

  hipStream_t stream = nullptr;

  constexpr int kStartPct = 1;
  constexpr int kEndPct = 50;
  constexpr int kStepPct = 2;
  constexpr int kMaxAllocs = (kEndPct - kStartPct) / kStepPct + 1;
  const std::size_t memLimit = (memBudget / 100) * 60;

  void* ptrs[kMaxAllocs] = {};
  std::size_t sizes[kMaxAllocs] = {};
  std::size_t expectedTotal = 0;

  // Allocate all blocks, stop when cumulative usage would exceed 60% of device memory
  int numAllocs = 0;
  for (int pct = kStartPct; pct <= kEndPct; pct += kStepPct) {
    std::size_t allocSize = (memBudget / 100) * pct;
    if (expectedTotal + allocSize > memLimit) break;
    sizes[numAllocs] = allocSize;
    expectedTotal += allocSize;
    HIP_CHECK(hipMallocAsync(&ptrs[numAllocs], sizes[numAllocs], stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    numAllocs++;
  }

  // Verify pool reports expected usage after all allocations
  uint64_t usedMem = 0;
  uint64_t reservedMem = 0;
  HIP_CHECK(hipMemPoolGetAttribute(pool, hipMemPoolAttrUsedMemCurrent, &usedMem));
  HIP_CHECK(hipMemPoolGetAttribute(pool, hipMemPoolAttrReservedMemCurrent, &reservedMem));
  REQUIRE(usedMem >= expectedTotal);
  REQUIRE(reservedMem >= expectedTotal);

  // Free all blocks
  for (int i = 0; i < numAllocs; i++) {
    HIP_CHECK(hipFreeAsync(ptrs[i], stream));
    HIP_CHECK(hipStreamSynchronize(stream));
  }

  // Verify pool reports zero usage after all frees
  HIP_CHECK(hipMemPoolGetAttribute(pool, hipMemPoolAttrUsedMemCurrent, &usedMem));
  HIP_CHECK(hipMemPoolGetAttribute(pool, hipMemPoolAttrReservedMemCurrent, &reservedMem));
  REQUIRE(usedMem == 0);
  REQUIRE(reservedMem == 0);
}
