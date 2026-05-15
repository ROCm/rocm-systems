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

#include <chrono>
#include <thread>

#include <hip_test_common.hh>
#include <performance_common.hh>
#include <resource_guards.hh>

/**
 * @addtogroup event event
 * @{
 * @ingroup PerformanceTest
 */

class HipEventElapsedTimeBenchmark : public Benchmark<HipEventElapsedTimeBenchmark> {
 public:
  void operator()() {
    hipEvent_t start, end;
    float time;

    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&end));

    HIP_CHECK(hipEventRecord(start));
    std::this_thread::sleep_for(std::chrono::milliseconds(5)); /* idle for 5 ms */
    HIP_CHECK(hipEventRecord(end));
    HIP_CHECK(hipEventSynchronize(end));

    TIMED_SECTION(kTimerTypeCpu) { HIP_CHECK(hipEventElapsedTime(&time, start, end)); }

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(end));
  }
};

/**
 * Test Description
 * ------------------------
 *  - Executes `hipEventElapsedTime`
 * Test source
 * ------------------------
 *  - performance/event/hipEventElapsedTime.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipEventElapsedTime") {
  HipEventElapsedTimeBenchmark benchmark;
  benchmark.Run();
}

/**
 * End doxygen group event.
 * @}
 */
