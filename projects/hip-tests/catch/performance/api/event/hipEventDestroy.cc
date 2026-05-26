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
 * @ingroup PerformanceTestEvent
 */

class HipEventDestroyBenchmark : public Benchmark<HipEventDestroyBenchmark> {
 public:
  void operator()() {
    hipEvent_t event;
    HIP_CHECK(hipEventCreate(&event));

    TIMED_SECTION(kTimerTypeCpu) { HIP_CHECK(hipEventDestroy(event)); }
  }
};

/**
 * Test Description
 * ------------------------
 *  - Executes `hipEventDestroy`
 * Test source
 * ------------------------
 * - performance/api/event/hipEventDestroy.cc
 * Test requirements
 * ------------------------
 * - performance/api/event/hipEventDestroy.cc
 */
HIP_TEST_CASE(Performance_hipEventDestroy) {
  HipEventDestroyBenchmark benchmark;
  benchmark.Run();
}

/**
 * End doxygen group event.
 * @}
 */
