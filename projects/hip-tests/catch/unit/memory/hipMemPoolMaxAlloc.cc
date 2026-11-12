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
  std::printf("hipMemGetInfo: free %zu total %zu\n", free, total);

  auto printPoolState = [&pool](const char* step, int alloc_idx, void* alloc_ptr,
                                std::size_t alloc_size) {
    uint64_t used_bytes = 0;
    uint64_t reserved_bytes = 0;
    HIP_CHECK(hipMemPoolGetAttribute(pool, hipMemPoolAttrUsedMemCurrent, &used_bytes));
    HIP_CHECK(hipMemPoolGetAttribute(pool, hipMemPoolAttrReservedMemCurrent, &reserved_bytes));
    std::printf("%s[%d] ptr %p size %zu: used %llu reserved %llu (%llu chunks of 128 MiB)\n", step,
                alloc_idx, alloc_ptr, alloc_size, static_cast<unsigned long long>(used_bytes),
                static_cast<unsigned long long>(reserved_bytes),
                static_cast<unsigned long long>(reserved_bytes >> 27));
    std::fflush(stdout);
  };

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
    printPoolState("alloc", numAllocs, ptrs[numAllocs], sizes[numAllocs]);
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
    printPoolState("free", i, ptrs[i], sizes[i]);
  }

  // Verify pool reports zero usage after all frees
  HIP_CHECK(hipMemPoolGetAttribute(pool, hipMemPoolAttrUsedMemCurrent, &usedMem));
  HIP_CHECK(hipMemPoolGetAttribute(pool, hipMemPoolAttrReservedMemCurrent, &reservedMem));
  REQUIRE(usedMem == 0);
  REQUIRE(reservedMem == 0);
}
