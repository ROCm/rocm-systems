/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
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
 * - 2) Allocate total 30GB of memory using hipMallocAsync in 2GB and 1GB chunks
 * - 3) Deallocate the 10 GB of memory out of 30 GB
 * - 4) Deallocate remaining 20 GB of memory
 * - 5) Print Free Device memory and Reserved and Used current memory details
 * -    after every step.
 * - 6) Performance improvement can be observed in terms of current used memory.
 *
 * Test source
 * ------------------------
 * - catch/perftests/memory/hipPerfMempool.cc
 *
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.5
 */

TEST_CASE("Perf_MempoolManager_hipMallocAsync_hipFreeAsync") {
  size_t free = 0, total = 0;
  HIP_CHECK(hipMemGetInfo(&free, &total));
  if (free < 30_GB) {
    HipTest::HIP_SKIP_TEST("Test requires 30 GB of device memory, skipping");
    return;
  }

  hipMemPool_t pool;
  int device = 0;
  HIP_CHECK(hipSetDevice(device));
  HIP_CHECK(hipDeviceGetDefaultMemPool(&pool, device));

  uint64_t threshold = 30_GB;
  HIP_CHECK(hipMemPoolSetAttribute(pool, hipMemPoolAttrReleaseThreshold, &threshold));

  std::cout << "\n Memory details at start : ";
  getAndPrintMemoryDetails(pool);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  constexpr int ptrs = 20;
  void* dPtr[ptrs];
  for (int i = 0; i < ptrs; i++) {
    dPtr[i] = nullptr;
  }

  // Allocate 30 GB (In 2GB and 1GB chunks)
  for (int i = 0; i < ptrs; i++) {
    if (i % 2 == 0) {
      HIP_CHECK(hipMallocAsync(&dPtr[i], 2_GB, stream));
    } else {
      HIP_CHECK(hipMallocAsync(&dPtr[i], 1_GB, stream));
    }
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  for (int i = 0; i < ptrs; i++) {
    REQUIRE(dPtr[i] != nullptr);
  }

  std::cout << "\n Memory details after allocating 30 GB memory : ";
  getAndPrintMemoryDetails(pool);

  // Free 10 GB out of 30 GB (All 1GB chunks)
  for (int i = 0; i < ptrs; i++) {
    if (i % 2 != 0) {
      HIP_CHECK(hipFreeAsync(dPtr[i], stream));
    }
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  std::cout << "\n Memory details after deallocating 10 GB out of 30 GB :";
  getAndPrintMemoryDetails(pool);

  // Free remaining 20 GB also (All 2GB chunks)
  for (int i = 0; i < ptrs; i++) {
    if (i % 2 == 0) {
      HIP_CHECK(hipFreeAsync(dPtr[i], stream));
    }
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  std::cout << "\n Memory details after deallocating the remaining 20 GB :";
  getAndPrintMemoryDetails(pool);

  HIP_CHECK(hipStreamDestroy(stream));
}
