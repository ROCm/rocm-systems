/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <performance_common.hh>
#include <resource_guards.hh>

/**
 * @addtogroup memset memset
 * @{
 * @ingroup PerformanceTest
 */

class Memset2DBenchmark : public Benchmark<Memset2DBenchmark> {
 public:
  Memset2DBenchmark(size_t width, size_t height) : dst_(width, height) {}

  void operator()() {
    TIMED_SECTION(kTimerTypeEvent) {
      HIP_CHECK(hipMemset2D(dst_.ptr(), dst_.pitch(), 17, dst_.width(), dst_.height()));
    }
  }

 private:
  LinearAllocGuard2D<char> dst_;
};

static void RunBenchmark(size_t width, size_t height) {
  Memset2DBenchmark benchmark(width, height);
  benchmark.AddSectionName("(" + std::to_string(width) + ", " + std::to_string(height) + ")");
  benchmark.Run();
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemset2D`:
 *    -# Allocation size
 *      - Small: 4 KB x 32 B
 *      - Medium: 4 MB x 32 B
 *      - Large: 16 MB x 32 B
 * Test source
 * ------------------------
 *  - performance/memset/hipMemset2D.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipMemset2D) {
  CHECK_IMAGE_SUPPORT
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(width, 32);
}

/**
 * End doxygen group PerformanceTest.
 * @}
 */
