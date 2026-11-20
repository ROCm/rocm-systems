/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_checkers.hh>
#include <performance_common.hh>

/**
 * The tests in this file is added to see the performance improvement with the
 * Design Memory manager for memory pool performance task : SWDEV-497841
 */

/*
 * Helper function to get and print the Total device and free device memory,
 * and reserved current, used current memory from pool.
 */
void getAndPrintMemoryDetails(const hipMemPool_t& pool) {
  size_t freeVRAM = 0, totalVRAM = 0, reservedCurrent = 0, usedCurrent = 0;

  HIP_CHECK(hipMemGetInfo(&freeVRAM, &totalVRAM));
  HIP_CHECK(hipMemPoolGetAttribute(pool, hipMemPoolAttrReservedMemCurrent, &reservedCurrent));
  HIP_CHECK(hipMemPoolGetAttribute(pool, hipMemPoolAttrUsedMemCurrent, &usedCurrent));

  std::cout << "\n Total device memory (GB) : " << totalVRAM / 1_GB;
  std::cout << "\n Free device memory (GB)  : " << freeVRAM / 1_GB;
  std::cout << "\n Pool Reserved current(GB): " << reservedCurrent / 1_GB;
  std::cout << "\n Pool Used current (GB)   : " << usedCurrent / 1_GB;
  std::cout << std::endl;
}

/**
 * Test Description
 * ------------------------
 * - This test case, tests the following scenario :
 * - 1) Get the default pool
 * - 2) Allocate twenty buffers in alternating two-unit and one-unit chunks.
 * -    Retain 1 GiB units where 30 GiB fits; otherwise use half of free memory.
 * - 3) Deallocate the ten one-unit buffers.
 * - 4) Deallocate the remaining ten two-unit buffers.
 * - 5) Print Free Device memory and Reserved and Used current memory details
 * -    after every step.
 * - 6) Performance improvement can be observed in terms of current used memory.
 *
 * Test source
 * ------------------------
 * - performance/scenarios/memory/hipPerfMempool.cc
 *
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.5
 */

HIP_TEST_CASE(Performance_MempoolManager_hipMallocAsync_hipFreeAsync) {
  constexpr int ptrs = 20;
  constexpr size_t allocation_units = (ptrs / 2) * 3;
  int device = 0;
  HIP_CHECK(hipSetDevice(device));

  size_t free = 0, total = 0;
  HIP_CHECK(hipMemGetInfo(&free, &total));
  // Preserve the original workload where 30 GiB fits. On smaller devices, leave
  // half of free memory for the runtime and pool allocation overhead, rounding
  // down so both chunk sizes are whole MiB.
  const size_t unit = free >= 30_GB ? 1_GB : (free / 2 / allocation_units) / 1_MB * 1_MB;
  if (unit == 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kNotEnoughFreeGpuMemory);
  }
  const size_t allocation_bytes = allocation_units * unit;
  std::cout << "Allocation unit (bytes): " << unit
            << ", total allocation (bytes): " << allocation_bytes << std::endl;

  hipMemPool_t pool;
  HIP_CHECK(hipDeviceGetDefaultMemPool(&pool, device));

  uint64_t threshold = allocation_bytes;
  HIP_CHECK(hipMemPoolSetAttribute(pool, hipMemPoolAttrReleaseThreshold, &threshold));

  std::cout << "\n Memory details at start : ";
  getAndPrintMemoryDetails(pool);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dPtr[ptrs];
  for (int i = 0; i < ptrs; i++) {
    dPtr[i] = nullptr;
  }

  // Allocate ten two-unit chunks and ten one-unit chunks.
  for (int i = 0; i < ptrs; i++) {
    if (i % 2 == 0) {
      HIP_CHECK(hipMallocAsync(&dPtr[i], 2 * unit, stream));
    } else {
      HIP_CHECK(hipMallocAsync(&dPtr[i], unit, stream));
    }
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  for (int i = 0; i < ptrs; i++) {
    REQUIRE(dPtr[i] != nullptr);
  }

  std::cout << "\n Memory details after allocating " << allocation_bytes << " bytes : ";
  getAndPrintMemoryDetails(pool);

  // Free the ten one-unit chunks.
  for (int i = 0; i < ptrs; i++) {
    if (i % 2 != 0) {
      HIP_CHECK(hipFreeAsync(dPtr[i], stream));
    }
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  std::cout << "\n Memory details after deallocating " << (ptrs / 2) * unit << " bytes out of "
            << allocation_bytes << " :";
  getAndPrintMemoryDetails(pool);

  // Free the remaining ten two-unit chunks.
  for (int i = 0; i < ptrs; i++) {
    if (i % 2 == 0) {
      HIP_CHECK(hipFreeAsync(dPtr[i], stream));
    }
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  std::cout << "\n Memory details after deallocating the remaining " << ptrs * unit << " bytes :";
  getAndPrintMemoryDetails(pool);

  HIP_CHECK(hipStreamDestroy(stream));
}
