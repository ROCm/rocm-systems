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

class Memset3DBenchmark : public Benchmark<Memset3DBenchmark> {
 public:
  Memset3DBenchmark(size_t width, size_t height, size_t depth) : dst_(width, height, depth) {}

  void operator()() {
    TIMED_SECTION(kTimerTypeEvent) {
      HIP_CHECK(hipMemset3D(dst_.pitched_ptr(), 17, dst_.extent()));
    }
  }

 private:
  LinearAllocGuard3D<char> dst_;
};

static void RunBenchmark(size_t width, size_t height, size_t depth) {
  Memset3DBenchmark benchmark(width, height, depth);
  benchmark.AddSectionName("(" + std::to_string(width) + ", " + std::to_string(height) + ", " +
                           std::to_string(depth) + ")");
  benchmark.Run();
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemset3D`:
 *    -# Allocation size
 *      - Small: 4 KB x 16 B x 4 B
 *      - Medium: 4 MB x 16 B x 4 B
 *      - Large: 16 MB x 16 B x 4 B
 * Test source
 * ------------------------
 *  - performance/memset/hipMemset3D.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Performance_hipMemset3D) {
  CHECK_IMAGE_SUPPORT
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(width, 16, 4);
}

/**
 * End doxygen group PerformanceTest.
 * @}
 */
