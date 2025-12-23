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

class Memset3DAsyncBenchmark : public Benchmark<Memset3DAsyncBenchmark> {
 public:
  Memset3DAsyncBenchmark(size_t width, size_t height, size_t depth)
      : dst_(width, height, depth), stream_(Streams::created) {}

  void operator()() {
    TIMED_SECTION_STREAM(kTimerTypeEvent, stream_.stream()) {
      HIP_CHECK(hipMemset3DAsync(dst_.pitched_ptr(), 17, dst_.extent(), stream_.stream()));
    }
    HIP_CHECK(hipStreamSynchronize(stream_.stream()));
  }

 private:
  LinearAllocGuard3D<char> dst_;
  StreamGuard stream_;
};

static void RunBenchmark(size_t width, size_t height, size_t depth) {
  Memset3DAsyncBenchmark benchmark(width, height, depth);
  benchmark.AddSectionName("(" + std::to_string(width) + ", " + std::to_string(height) + ", " +
                           std::to_string(depth) + ")");
  benchmark.Run();
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemset3DAsync`:
 *    -# Allocation size
 *      - Small: 4 KB x 16 B x 4 B
 *      - Medium: 4 MB x 16 B x 4 B
 *      - Large: 16 MB x 16 B x 4 B
 * Test source
 * ------------------------
 *  - performance/memset/hipMemset3DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemset3DAsync") {
  CHECK_IMAGE_SUPPORT
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(width, 16, 4);
}

/**
 * End doxygen group PerformanceTest.
 * @}
 */
