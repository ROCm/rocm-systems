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

class StreamGetFlagsBenchmark : public Benchmark<StreamGetFlagsBenchmark> {
 public:
  void operator()(unsigned int expected_flag) {
    unsigned int returned_flags{};
    hipStream_t stream;

    HIP_CHECK(hipStreamCreateWithFlags(&stream, expected_flag));
    TIMED_SECTION(kTimerTypeCpu){HIP_CHECK(hipStreamGetFlags(stream, &returned_flags))} HIP_CHECK(
        hipStreamDestroy(stream));
  }
};

static void RunBenchmark(unsigned int expected_flag) {
  StreamGetFlagsBenchmark benchmark;
  switch (expected_flag) {
    case hipStreamDefault:
      benchmark.AddSectionName("hipStreamDefault");
      break;
    case hipStreamNonBlocking:
      benchmark.AddSectionName("hipStreamNonBlocking");
      break;
    default:
      benchmark.AddSectionName("unknown flag type");
  }
  benchmark.Run(expected_flag);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipStreamGetFlags`:
 *    -# Flags:
 *      - `hipStreamDefault`
 *      - `hipStreamNonBlocking`
 * Test source
 * ------------------------
 *  - performance/stream/hipStreamGetFlags.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipStreamGetFlags) {
  unsigned int expected_flag = GENERATE(hipStreamDefault, hipStreamNonBlocking);
  RunBenchmark(expected_flag);
}

/**
 * End doxygen group PerformanceTest.
 * @}
 */
