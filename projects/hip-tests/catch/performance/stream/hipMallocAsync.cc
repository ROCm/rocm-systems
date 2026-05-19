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

class MallocAsyncBenchmark : public Benchmark<MallocAsyncBenchmark> {
 public:
  void operator()(const size_t array_size) {
    const StreamGuard stream_guard{Streams::created};
    const hipStream_t stream = stream_guard.stream();
    float* dev_ptr{nullptr};

    TIMED_SECTION_STREAM(kTimerTypeEvent, stream) {
      HIP_CHECK(
          hipMallocAsync(reinterpret_cast<void**>(&dev_ptr), array_size * sizeof(float), stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(dev_ptr));
  }
};

static void RunBenchmark(const size_t array_size) {
  MallocAsyncBenchmark benchmark;
  benchmark.AddSectionName(std::to_string(array_size));
  benchmark.Run(array_size);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMallocAsync` with created stream:
 *    -# Allocation size:
 *      - 4 KB
 *      - 4 MB
 *      - 16 MB
 * Test source
 * ------------------------
 *  - performance/stream/hipMallocAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipMallocAsync) {
  size_t array_size = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(array_size);
}

/**
 * End doxygen group PerformanceTest.
 * @}
 */
