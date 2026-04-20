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

class ExtStreamGetCUMaskBenchmark : public Benchmark<ExtStreamGetCUMaskBenchmark> {
 public:
  void operator()() {
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    std::vector<uint32_t> cu_mask(props.multiProcessorCount, 0);
    hipStream_t stream{};
    HIP_CHECK(hipExtStreamCreateWithCUMask(&stream, cu_mask.size(), cu_mask.data()));
    std::vector<uint32_t> new_cu_mask(cu_mask.size(), 0);

    TIMED_SECTION(kTimerTypeCpu) {
      HIP_CHECK(hipExtStreamGetCUMask(stream, new_cu_mask.size(), new_cu_mask.data()));
    }

    HIP_CHECK(hipStreamDestroy(stream));
  }
};

static void RunBenchmark() {
  ExtStreamGetCUMaskBenchmark benchmark;
  benchmark.Run();
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipExtStreamGetCUMask`.
 *  - Creates basic mask and gets it into the new one.
 * Test source
 * ------------------------
 *  - performance/stream/hipExtStreamGetCUMask.cc
 * Test requirements
 * ------------------------
 *  - Platform specific (AMD)
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipExtStreamGetCUMask) { RunBenchmark(); }

/**
 * End doxygen group PerformanceTest.
 * @}
 */
