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

class MemPoolSetAccessBenchmark : public Benchmark<MemPoolSetAccessBenchmark> {
 public:
  void operator()() {
    hipMemPool_t mem_pool{nullptr};
    hipMemPoolProps pool_props = CreateMemPoolProps(0, hipMemHandleTypeNone);
    HIP_CHECK(hipMemPoolCreate(&mem_pool, &pool_props));

    hipMemAccessDesc desc_list = {{hipMemLocationTypeDevice, 0}, hipMemAccessFlagsProtReadWrite};

    TIMED_SECTION(kTimerTypeCpu) { HIP_CHECK(hipMemPoolSetAccess(mem_pool, &desc_list, 1)); }

    HIP_CHECK(hipMemPoolDestroy(mem_pool));
  }
};

static void RunBenchmark() {
  MemPoolSetAccessBenchmark benchmark;
  benchmark.Run();
}

/**
 * @warning **MemPool APIs are not fully implemented within current version
 *          or HIP and therefore they cannot be appropriately executed on AMD and NVIDIA platforms.
 *          Therefore, all tests related to MemPool APIs are implemented without formal
 *          verification and will be verified once HIP fully supports MemPool APIs.**
 * Test Description
 * ------------------------
 *  - Executes `hipMemPoolSetAccess` with `hipMemAccessFlagsProtReadWrite`.
 * Test source
 * ------------------------
 *  - performance/stream/hipMemPoolSetAccess.cc
 * Test requirements
 * ------------------------
 *  - Device supports memory pools
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Performance_hipMemPoolSetAccess) {
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
