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

void Callback(hipStream_t stream, hipError_t status, void* user_data) {}

class StreamAddCallbackBenchmark : public Benchmark<StreamAddCallbackBenchmark> {
 public:
  void operator()() {
    const StreamGuard stream_guard{Streams::created};
    const hipStream_t stream = stream_guard.stream();

    TIMED_SECTION(kTimerTypeCpu) { HIP_CHECK(hipStreamAddCallback(stream, Callback, nullptr, 0)); }
  }
};

static void RunBenchmark() {
  StreamAddCallbackBenchmark benchmark;
  benchmark.Run();
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipStreamAddCallback` on the created stream.
 * Test source
 * ------------------------
 *  - performance/stream/hipStreamAddCallback.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipStreamAddCallback) { RunBenchmark(); }

/**
 * End doxygen group PerformanceTest.
 * @}
 */
