/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <performance_common.hh>

/**
 * @addtogroup stream stream
 * @{
 * @ingroup PerformanceTest
 */

class StreamGetPriorityBenchmark : public Benchmark<StreamGetPriorityBenchmark> {
 public:
  void operator()(Streams stream_type) {
    const StreamGuard stream_guard{stream_type};
    const hipStream_t stream = stream_guard.stream();

    int priority{};
    TIMED_SECTION(kTimerTypeCpu) { HIP_CHECK(hipStreamGetPriority(stream, &priority)); }
  }
};

static void RunBenchmark(Streams stream_type) {
  StreamGetPriorityBenchmark benchmark;
  switch (stream_type) {
    case Streams::nullstream:
      benchmark.AddSectionName("null stream");
      break;
    case Streams::created:
      benchmark.AddSectionName("created");
      break;
    default:
      benchmark.AddSectionName("per thread stream");
  }
  benchmark.Run(stream_type);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipStreamGetPriority`:
 *    -# Stream types:
 *      - `null`
 *      - created
 * Test source
 * ------------------------
 *  - performance/stream/hipStreamGetPriority.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Performance_hipStreamGetPriority) {
  Streams stream_type = GENERATE(Streams::nullstream, Streams::created);
  RunBenchmark(stream_type);
}

/**
 * End doxygen group PerformanceTest.
 * @}
 */
