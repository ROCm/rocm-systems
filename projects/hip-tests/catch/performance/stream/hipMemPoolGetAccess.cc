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

class MemPoolGetAccessBenchmark : public Benchmark<MemPoolGetAccessBenchmark> {
 public:
  void operator()() {
    hipMemPool_t mem_pool{nullptr};
    hipMemPoolProps pool_props = CreateMemPoolProps(0, hipMemHandleTypeNone);
    HIP_CHECK(hipMemPoolCreate(&mem_pool, &pool_props));

    hipMemAccessFlags flags = hipMemAccessFlagsProtNone;
    hipMemLocation location = {hipMemLocationTypeDevice, 0};
    TIMED_SECTION(kTimerTypeCpu) { HIP_CHECK(hipMemPoolGetAccess(&flags, mem_pool, location)); }

    HIP_CHECK(hipMemPoolDestroy(mem_pool));
  }
};

static void RunBenchmark() {
  MemPoolGetAccessBenchmark benchmark;
  benchmark.Run();
}

/**
 * @warning **MemPool APIs are not fully implemented within current version
 *          or HIP and therefore they cannot be appropriately executed on AMD and NVIDIA platforms.
 *          Therefore, all tests related to MemPool APIs are implemented without formal
 *          verification and will be verified once HIP fully supports MemPool APIs.**
 * Test Description
 * ------------------------
 *  - Executes `hipMemPoolGetAccess`.
 * Test source
 * ------------------------
 *  - performance/stream/hipMemPoolGetAccess.cc
 * Test requirements
 * ------------------------
 *  - Device supports memory pools
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipMemPoolGetAccess) {
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
