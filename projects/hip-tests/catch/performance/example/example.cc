/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <performance_common.hh>
#include <resource_guards.hh>

class ExampleBenchmark : public Benchmark<ExampleBenchmark> {
 public:
  void operator()(void* dst) {
    const int value = 42;
    const size_t kSize = 4_MB;

    TIMED_SECTION(kTimerTypeEvent) {  // event based timing
      HIP_CHECK(hipMemset(dst, value, kSize));
    }

    HIP_CHECK(hipMemset(dst, 0, kSize));  // not timed

    TIMED_SECTION(kTimerTypeCpu) {  // cpu based timing
      HIP_CHECK(hipMemset(dst, value, kSize));
    }

    // accessing properties
    // std::cout << "Time recorded up until now: " << time() << std::endl;
    // std::cout << "Number of iterations: " << iterations() << std::endl;
    // std::cout << "Number of warmup iterations: " << warmups() << std::endl;
    // std::cout << "Current iteration: " << current() << std::endl;
  }
};

TEST_CASE(Performance_Example) {
  ExampleBenchmark benchmark;

  // to override cmd options
  // benchmark.Configure(10000 /* iterations */, 1000 /* warmups */);

  LinearAllocGuard<void> dst(LinearAllocs::hipMalloc, 4_MB);
  benchmark.Run(dst.ptr());
}
