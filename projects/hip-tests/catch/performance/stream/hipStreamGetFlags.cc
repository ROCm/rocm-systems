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
TEST_CASE("Performance_hipStreamGetFlags") {
  unsigned int expected_flag = GENERATE(hipStreamDefault, hipStreamNonBlocking);
  RunBenchmark(expected_flag);
}

/**
 * End doxygen group PerformanceTest.
 * @}
 */
