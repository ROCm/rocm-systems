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

class MemPoolTrimToBenchmark : public Benchmark<MemPoolTrimToBenchmark> {
 public:
  void operator()(const size_t min_bytes_to_hold) {
    hipMemPool_t mem_pool{nullptr};
    hipMemPoolProps pool_props = CreateMemPoolProps(0, hipMemHandleTypeNone);
    HIP_CHECK(hipMemPoolCreate(&mem_pool, &pool_props));

    TIMED_SECTION(kTimerTypeCpu) { HIP_CHECK(hipMemPoolTrimTo(mem_pool, min_bytes_to_hold)); }

    HIP_CHECK(hipMemPoolDestroy(mem_pool));
  }
};

static void RunBenchmark(const size_t min_bytes_to_hold) {
  MemPoolTrimToBenchmark benchmark;
  benchmark.AddSectionName(std::to_string(min_bytes_to_hold));
  benchmark.Run(min_bytes_to_hold);
}

/**
 * @warning **MemPool APIs are not fully implemented within current version
 *          or HIP and therefore they cannot be appropriately executed on AMD and NVIDIA platforms.
 *          Therefore, all tests related to MemPool APIs are implemented without formal
 *          verification and will be verified once HIP fully supports MemPool APIs.**
 * Test Description
 * ------------------------
 *  - Executes `hipMemPoolTrimTo`:
 *    -# Minimum bytes to hold:
 *      - 4 KB
 *      - 4 MB
 *      - 16 MB
 * Test source
 * ------------------------
 *  - performance/stream/hipMemPoolTrimTo.cc
 * Test requirements
 * ------------------------
 *  - Device supports memory pools
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipMemPoolTrimTo) {
  if (!AreMemPoolsSupported(0)) {
    HipTest::HIP_SKIP_TEST(
        "GPU 0 doesn't support hipDeviceAttributeMemoryPoolsSupported "
        "attribute. Hence skipping the testing with Pass result.\n");
    return;
  }
  size_t min_bytes_to_hold = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(min_bytes_to_hold);
}

/**
 * End doxygen group PerformanceTest.
 * @}
 */
