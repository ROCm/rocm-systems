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

class StreamWaitEventBenchmark : public Benchmark<StreamWaitEventBenchmark> {
 public:
  void operator()(Streams stream_type) {
    const StreamGuard stream_guard{stream_type};
    const hipStream_t stream = stream_guard.stream();
    hipEvent_t wait_event{nullptr};

    HIP_CHECK(hipEventCreate(&wait_event));
    REQUIRE(wait_event != nullptr);
    HIP_CHECK(hipEventRecord(wait_event, stream));

    TIMED_SECTION(kTimerTypeCpu) {
      HIP_CHECK(hipStreamWaitEvent(stream, wait_event, 0));
      HIP_CHECK(hipStreamSynchronize(stream));
    }
    HIP_CHECK(hipEventDestroy(wait_event));
  }
};

static void RunBenchmark(Streams stream_type) {
  StreamWaitEventBenchmark benchmark{};
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
 *  - Executes `hipStreamWaitEvent`:
 *    -# Stream types:
 *      - `null`
 *      - created
 * Test source
 * ------------------------
 *  - performance/stream/hipStreamWaitEvent.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipStreamWaitEvent") {
  Streams stream_type = GENERATE(Streams::nullstream, Streams::created);
  RunBenchmark(stream_type);
}

/**
 * End doxygen group PerformanceTest.
 * @}
 */
