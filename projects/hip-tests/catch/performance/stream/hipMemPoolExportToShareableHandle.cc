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

class MemPoolExportToShareableHandleBenchmark
    : public Benchmark<MemPoolExportToShareableHandleBenchmark> {
 public:
  void operator()() {
    hipMemPool_t mem_pool{nullptr};
    int share_handle;

    hipMemPoolProps props = CreateMemPoolProps(0, kHandleType);
    HIP_CHECK(hipMemPoolCreate(&mem_pool, &props));

    TIMED_SECTION(kTimerTypeCpu) {
      HIP_CHECK(hipMemPoolExportToShareableHandle(&share_handle, mem_pool, kHandleType, 0));
    }

    HIP_CHECK(hipMemPoolDestroy(mem_pool));
  }
};

static void RunBenchmark() {
  MemPoolExportToShareableHandleBenchmark benchmark;
  benchmark.Run();
}

/**
 * @warning **MemPool APIs are not fully implemented within current version
 *          or HIP and therefore they cannot be appropriately executed on AMD and NVIDIA platforms.
 *          Therefore, all tests related to MemPool APIs are implemented without formal
 *          verification and will be verified once HIP fully supports MemPool APIs.**
 * Test Description
 * ------------------------
 *  - Executes `hipMemPoolExportToShareableHandle`.
 *  - Uses the same process for import and export operations.
 * Test source
 * ------------------------
 *  - performance/stream/hipMemPoolExportToShareableHandle.cc
 * Test requirements
 * ------------------------
 *  - Device supports memory pools
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipMemPoolExportToShareableHandle) {
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
