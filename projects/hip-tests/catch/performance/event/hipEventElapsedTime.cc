/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <chrono>
#include <thread>

#include <hip_test_common.hh>
#include <performance_common.hh>
#include <resource_guards.hh>

/**
 * @addtogroup event event
 * @{
 * @ingroup PerformanceTest
 */

class HipEventElapsedTimeBenchmark : public Benchmark<HipEventElapsedTimeBenchmark> {
 public:
  void operator()() {
    hipEvent_t start, end;
    float time;

    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&end));

    HIP_CHECK(hipEventRecord(start));
    std::this_thread::sleep_for(std::chrono::milliseconds(5)); /* idle for 5 ms */
    HIP_CHECK(hipEventRecord(end));
    HIP_CHECK(hipEventSynchronize(end));

    TIMED_SECTION(kTimerTypeCpu) { HIP_CHECK(hipEventElapsedTime(&time, start, end)); }

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(end));
  }
};

/**
 * Test Description
 * ------------------------
 *  - Executes `hipEventElapsedTime`
 * Test source
 * ------------------------
 *  - performance/event/hipEventElapsedTime.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Performance_hipEventElapsedTime) {
  HipEventElapsedTimeBenchmark benchmark;
  benchmark.Run();
}

/**
 * End doxygen group event.
 * @}
 */
