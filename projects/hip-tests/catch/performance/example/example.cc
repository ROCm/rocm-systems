/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
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

TEST_CASE("Performance_Example") {
  ExampleBenchmark benchmark;

  // to override cmd options
  // benchmark.Configure(10000 /* iterations */, 1000 /* warmups */);

  LinearAllocGuard<void> dst(LinearAllocs::hipMalloc, 4_MB);
  benchmark.Run(dst.ptr());
}
