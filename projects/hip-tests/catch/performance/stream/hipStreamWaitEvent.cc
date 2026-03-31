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

class StreamWaitEventBenchmark : public Benchmark<StreamWaitEventBenchmark> {
 public:
  void operator()(Streams stream_type) {
    const StreamGuard stream_guard{stream_type};
    const hipStream_t stream = stream_guard.stream();
    hipEvent_t wait_event{nullptr};

    HIP_CHECK(hipEventCreate(&wait_event));
    REQUIRE(wait_event != nullptr);
    HIP_CHECK(hipEventRecord(wait_event, stream));

    TIMED_SECTION(kTimerTypeCpu) {
      HIP_CHECK(hipStreamWaitEvent(stream, wait_event, 0));
      HIP_CHECK(hipStreamSynchronize(stream));
    }
    HIP_CHECK(hipEventDestroy(wait_event));
  }
};

static void RunBenchmark(Streams stream_type) {
  StreamWaitEventBenchmark benchmark{};
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
 *  - Executes `hipStreamWaitEvent`:
 *    -# Stream types:
 *      - `null`
 *      - created
 * Test source
 * ------------------------
 *  - performance/stream/hipStreamWaitEvent.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipStreamWaitEvent) {
  Streams stream_type = GENERATE(Streams::nullstream, Streams::created);
  RunBenchmark(stream_type);
}

/**
 * End doxygen group PerformanceTest.
 * @}
 */
