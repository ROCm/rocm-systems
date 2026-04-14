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

class MemPoolCreateBenchmark : public Benchmark<MemPoolCreateBenchmark> {
 public:
  void operator()() {
    hipMemPool_t mem_pool{nullptr};
    hipMemPoolProps pool_props = CreateMemPoolProps(0, hipMemHandleTypeNone);

    TIMED_SECTION(kTimerTypeCpu) { HIP_CHECK(hipMemPoolCreate(&mem_pool, &pool_props)); }

    REQUIRE(mem_pool != nullptr);
    HIP_CHECK(hipMemPoolDestroy(mem_pool));
  }
};

static void RunBenchmark() {
  MemPoolCreateBenchmark benchmark;
  benchmark.Run();
}

/**
 * @warning **MemPool APIs are not fully implemented within current version
 *          or HIP and therefore they cannot be appropriately executed on AMD and NVIDIA platforms.
 *          Therefore, all tests related to MemPool APIs are implemented without formal
 *          verification and will be verified once HIP fully supports MemPool APIs.**
 * Test Description
 * ------------------------
 *  - Executes `hipMemPoolCreate`.
 * Test source
 * ------------------------
 *  - performance/stream/hipMemPoolCreate.cc
 * Test requirements
 * ------------------------
 *  - Device supports memory pools
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Performance_hipMemPoolCreate) {
  if (!AreMemPoolsSupported(0)) {
    HipTest::HIP_SKIP_TEST(
        "GPU 0 doesn't support hipDeviceAttributeMemoryPoolsSupported "
        "attribute. Hence skipping the testing with Pass result.\n");
    return;
  }
  RunBenchmark();
}

/**
 * End doxygen group PerformanceTest.
 * @}
 */
