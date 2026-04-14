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

class FreeAsyncBenchmark : public Benchmark<FreeAsyncBenchmark> {
 public:
  void operator()(const size_t array_size) {
    const StreamGuard stream_guard{Streams::created};
    const hipStream_t stream = stream_guard.stream();
    float* dev_ptr{nullptr};
    HIP_CHECK(
        hipMallocAsync(reinterpret_cast<void**>(&dev_ptr), array_size * sizeof(float), stream));

    TIMED_SECTION_STREAM(kTimerTypeEvent, stream) { HIP_CHECK(hipFreeAsync(dev_ptr, stream)); }

    HIP_CHECK(hipStreamSynchronize(stream));
  }
};

static void RunBenchmark(const size_t array_size) {
  FreeAsyncBenchmark benchmark;
  benchmark.AddSectionName(std::to_string(array_size));
  benchmark.Run(array_size);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipFreeAsync` with created stream:
 *    -# Allocation size:
 *      - 4 KB
 *      - 4 MB
 *      - 16 MB
 * Test source
 * ------------------------
 *  - performance/stream/hipFreeAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Performance_hipFreeAsync) {
  size_t array_size = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(array_size);
}

/**
 * End doxygen group PerformanceTest.
 * @}
 */
