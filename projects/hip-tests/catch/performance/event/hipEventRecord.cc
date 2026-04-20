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

class HipEventRecordBenchmark : public Benchmark<HipEventRecordBenchmark> {
 public:
  void operator()(hipStream_t stream) {
    hipEvent_t event;
    HIP_CHECK(hipEventCreate(&event));

    TIMED_SECTION(kTimerTypeCpu) { HIP_CHECK(hipEventRecord(event, stream)); }

    HIP_CHECK(hipEventDestroy(event));
  }
};

static void RunBenchmark(hipStream_t stream) {
  HipEventRecordBenchmark benchmark;
  if (stream == NULL) {
    benchmark.AddSectionName("Default stream");
  } else {
    benchmark.AddSectionName("Created stream");
  }
  benchmark.Run(stream);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipEventRecord`
 *    -# Executed both on
 *      - default stream
 *      - created stream
 * Test source
 * ------------------------
 *  - performance/event/hipEventRecord.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipEventRecord) {
  SECTION("default stream") { RunBenchmark(nullptr); }

  SECTION("created stream") {
    StreamGuard stream(Streams::created);
    RunBenchmark(stream.stream());
  }
}

/**
 * End doxygen group event.
 * @}
 */
