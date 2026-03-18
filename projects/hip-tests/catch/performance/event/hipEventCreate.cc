/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <performance_common.hh>
#include <resource_guards.hh>

/**
 * @addtogroup event event
 * @{
 * @ingroup PerformanceTest
 * Contains performance tests for all hipEvent related HIP APIs.
 */

class HipEventCreateBenchmark : public Benchmark<HipEventCreateBenchmark> {
 public:
  void operator()() {
    hipEvent_t event;

    TIMED_SECTION(kTimerTypeCpu) { HIP_CHECK(hipEventCreate(&event)); }

    HIP_CHECK(hipEventDestroy(event));
  }
};

/**
 * Test Description
 * ------------------------
 *  - Executes `hipEventCreate`
 * Test source
 * ------------------------
 *  - performance/event/hipEventCreate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Performance_hipEventCreate) {
  HipEventCreateBenchmark benchmark;
  benchmark.Run();
}

/**
 * End doxygen group PerformanceTest.
 * @}
 */
