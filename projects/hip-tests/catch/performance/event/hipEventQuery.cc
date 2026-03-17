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
 */

class HipEventQueryBenchmark : public Benchmark<HipEventQueryBenchmark> {
 public:
  void operator()() {
    hipEvent_t event;
    HIP_CHECK(hipEventCreate(&event));
    HIP_CHECK(hipEventRecord(event));
    HIP_CHECK(hipEventSynchronize(event));

    TIMED_SECTION(kTimerTypeCpu) { HIP_CHECK(hipEventQuery(event)); }

    HIP_CHECK(hipEventDestroy(event));
  }
};

/**
 * Test Description
 * ------------------------
 *  - Executes `hipEventQuery`
 * Test source
 * ------------------------
 *  - performance/event/hipEventQuery.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Performance_hipEventQuery) {
  HipEventQueryBenchmark benchmark;
  benchmark.Run();
}

/**
 * End doxygen group event.
 * @}
 */
