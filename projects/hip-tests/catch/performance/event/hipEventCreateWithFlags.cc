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

class HipEventCreateWithFlagsBenchmark : public Benchmark<HipEventCreateWithFlagsBenchmark> {
 public:
  void operator()(unsigned flag) {
    hipEvent_t event;

    TIMED_SECTION(kTimerTypeCpu) { HIP_CHECK(hipEventCreateWithFlags(&event, flag)); }

    HIP_CHECK(hipEventDestroy(event));
  }
};

static std::string GetEventCreateFlagName(unsigned flag) {
  switch (flag) {
    case hipEventDefault:
      return "hipEventDefault";
    case hipEventBlockingSync:
      return "hipEventBlockingSync";
    case hipEventDisableTiming:
      return "hipEventDisableTiming";
    case hipEventInterprocess:
      return "hipEventInterprocess";
    default:
      return "flag combination";
  }
}

static void RunBenchmark(unsigned flag) {
  HipEventCreateWithFlagsBenchmark benchmark;
  benchmark.AddSectionName(GetEventCreateFlagName(flag));
  benchmark.Run(flag);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipEventCreateWithFlags` with all flags:
 *    -# Flags
 *      - hipEventDefault
 *      - hipEventBlockingSync
 *      - hipEventDisableTiming
 *      - hipEventInterprocess (currently disabled)
 * Test source
 * ------------------------
 *  - performance/event/hipEventCreate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipEventCreateWithFlags) {
  const auto flag = GENERATE(
      hipEventDefault, hipEventBlockingSync,
      hipEventDisableTiming /*, hipEventInterprocess  disabled until fixed (EXSWHTEC-25) */);
  RunBenchmark(flag);
}

/**
 * End doxygen group PerformanceTest.
 * @}
 */
