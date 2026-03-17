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

class HipEventSynchronizeBenchmark : public Benchmark<HipEventSynchronizeBenchmark> {
 public:
  void operator()(unsigned flag) {
    hipEvent_t event;
    HIP_CHECK(hipEventCreateWithFlags(&event, flag));
    HIP_CHECK(hipEventRecord(event));

    TIMED_SECTION(kTimerTypeCpu) { HIP_CHECK(hipEventSynchronize(event)); }

    HIP_CHECK(hipEventDestroy(event));
  }
};

static void RunBenchmark(unsigned flag) {
  HipEventSynchronizeBenchmark benchmark;
  if (flag == hipEventDefault) {
    benchmark.AddSectionName("Default event");
  } else {
    benchmark.AddSectionName("Blocking sync event");
  }
  benchmark.Run(flag);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipEventSynchronize`
 *    -# Checked on events created with flags:
 *      - hipEventDefault
 *      - hipEventBlockingSync
 * Test source
 * ------------------------
 *  - performance/event/hipEventSynchronize.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipEventSynchronize) {
  const auto flag = GENERATE(hipEventDefault, hipEventBlockingSync);
  RunBenchmark(flag);
}

/**
 * End doxygen group event.
 * @}
 */
