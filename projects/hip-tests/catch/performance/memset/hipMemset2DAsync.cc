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

/**
 * @addtogroup memset memset
 * @{
 * @ingroup PerformanceTest
 */

class Memset2DAsyncBenchmark : public Benchmark<Memset2DAsyncBenchmark> {
 public:
  Memset2DAsyncBenchmark(size_t width, size_t height)
      : dst_(width, height), stream_(Streams::created) {}

  void operator()(size_t width, size_t height) {
    TIMED_SECTION_STREAM(kTimerTypeEvent, stream_.stream()) {
      HIP_CHECK(hipMemset2DAsync(dst_.ptr(), dst_.pitch(), 17, dst_.width(), dst_.height(),
                                 stream_.stream()));
    }
    HIP_CHECK(hipStreamSynchronize(stream_.stream()));
  }

 private:
  LinearAllocGuard2D<char> dst_;
  StreamGuard stream_;
};

static void RunBenchmark(size_t width, size_t height) {
  Memset2DAsyncBenchmark benchmark(width, height);
  benchmark.AddSectionName("(" + std::to_string(width) + ", " + std::to_string(height) + ")");
  benchmark.Run(width, height);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemset2DAsync`:
 *    -# Allocation size
 *      - Small: 4 KB x 32 B
 *      - Medium: 4 MB x 32 B
 *      - Large: 16 MB x 32 B
 * Test source
 * ------------------------
 *  - performance/memset/hipMemset2DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemset2DAsync") {
  CHECK_IMAGE_SUPPORT
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(width, 32);
}

/**
 * End doxygen group PerformanceTest.
 * @}
 */
