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

class StreamGetPriorityBenchmark : public Benchmark<StreamGetPriorityBenchmark> {
 public:
  void operator()(Streams stream_type) {
    const StreamGuard stream_guard{stream_type};
    const hipStream_t stream = stream_guard.stream();

    int priority{};
    TIMED_SECTION(kTimerTypeCpu) { HIP_CHECK(hipStreamGetPriority(stream, &priority)); }
  }
};

static void RunBenchmark(Streams stream_type) {
  StreamGetPriorityBenchmark benchmark;
  switch (stream_type) {
    case Streams::nullstream:
      benchmark.AddSectionName("null stream");
      break;
    case Streams::created:
      benchmark.AddSectionName("created");
      break;
    default:
      benchmark.AddSectionName("per thread stream");
  }
  benchmark.Run(stream_type);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipStreamGetPriority`:
 *    -# Stream types:
 *      - `null`
 *      - created
 * Test source
 * ------------------------
 *  - performance/stream/hipStreamGetPriority.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipStreamGetPriority") {
  Streams stream_type = GENERATE(Streams::nullstream, Streams::created);
  RunBenchmark(stream_type);
}

/**
 * End doxygen group PerformanceTest.
 * @}
 */
