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
 * Contains performance tests for all stream management HIP APIs.
 */

class ExtStreamCreateWithCUMaskBenchmark : public Benchmark<ExtStreamCreateWithCUMaskBenchmark> {
 public:
  void operator()() {
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    std::vector<uint32_t> cu_mask(props.multiProcessorCount, 0);
    hipStream_t stream{};

    TIMED_SECTION(kTimerTypeCpu) {
      HIP_CHECK(hipExtStreamCreateWithCUMask(&stream, cu_mask.size(), cu_mask.data()));
    }

    HIP_CHECK(hipStreamDestroy(stream));
  }
};

static void RunBenchmark() {
  ExtStreamCreateWithCUMaskBenchmark benchmark;
  benchmark.Run();
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipExtStreamCreateWithCUMask`.
 * Test source
 * ------------------------
 *  - performance/stream/hipExtStreamCreateWithCUMask.cc
 * Test requirements
 * ------------------------
 *  - Platform specific (AMD)
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipExtStreamCreateWithCUMask) { RunBenchmark(); }

/**
 * End doxygen group PerformanceTest.
 * @}
 */
