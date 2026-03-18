/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "mem_pools_performance_common.hh"

/**
 * @addtogroup stream stream
 * @{
 * @ingroup PerformanceTest
 */

class MemPoolImportPointerBenchmark : public Benchmark<MemPoolImportPointerBenchmark> {
 public:
  void operator()(const size_t array_size) {
    float* device_ptr{nullptr};
    float* device_ptr_import{nullptr};
    hipMemPool_t mem_pool{nullptr};
    hipMemPoolPtrExportData exp_data;

    hipMemPoolProps props = CreateMemPoolProps(0, kHandleType);
    HIP_CHECK(hipMemPoolCreate(&mem_pool, &props));
    HIP_CHECK(hipMallocFromPoolAsync(&device_ptr, array_size * sizeof(float), mem_pool, nullptr));
    HIP_CHECK(hipStreamSynchronize(nullptr));
    HIP_CHECK(hipMemPoolExportPointer(&exp_data, device_ptr));

    TIMED_SECTION(kTimerTypeCpu) {
      HIP_CHECK(hipMemPoolImportPointer(reinterpret_cast<void**>(&device_ptr_import), mem_pool,
                                        &exp_data));
    }

    HIP_CHECK(hipFree(device_ptr));
    HIP_CHECK(hipFree(device_ptr_import));
    HIP_CHECK(hipMemPoolDestroy(mem_pool));
  }
};

static void RunBenchmark(const size_t array_size) {
  MemPoolImportPointerBenchmark benchmark;
  benchmark.AddSectionName(std::to_string(array_size));
  benchmark.Run(array_size);
}

/**
 * @warning **MemPool APIs are not fully implemented within current version
 *          or HIP and therefore they cannot be appropriately executed on AMD and NVIDIA platforms.
 *          Therefore, all tests related to MemPool APIs are implemented without formal
 *          verification and will be verified once HIP fully supports MemPool APIs.**
 * Test Description
 * ------------------------
 *  - Executes `hipMemPoolImportPointer`:
 *    -# Allocation size:
 *      - 4 KB
 *      - 4 MB
 *      - 16 MB
 *  - Uses the same process for import and export operations.
 * Test source
 * ------------------------
 *  - performance/stream/hipMemPoolImportPointer.cc
 * Test requirements
 * ------------------------
 *  - Device supports memory pools
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Performance_hipMemPoolImportPointer) {
  if (!AreMemPoolsSupported(0)) {
    HipTest::HIP_SKIP_TEST(
        "GPU 0 doesn't support hipDeviceAttributeMemoryPoolsSupported "
        "attribute. Hence skipping the testing with Pass result.\n");
    return;
  }
  size_t array_size = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(array_size);
}

/**
 * End doxygen group PerformanceTest.
 * @}
 */
