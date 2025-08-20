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
 * @addtogroup event event
 * @{
 * @ingroup PerformanceTest
 */

class HipEventCreateWithFlagsBenchmark : public Benchmark<HipEventCreateWithFlagsBenchmark> {
 public:
  void operator()(unsigned flag) {
    hipEvent_t event;

    TIMED_SECTION(kTimerTypeCpu) { HIP_CHECK(hipEventCreateWithFlags(&event, flag)); }

    HIP_CHECK(hipEventDestroy(event));
  }
};

static std::string GetEventCreateFlagName(unsigned flag) {
  switch (flag) {
    case hipEventDefault:
      return "hipEventDefault";
    case hipEventBlockingSync:
      return "hipEventBlockingSync";
    case hipEventDisableTiming:
      return "hipEventDisableTiming";
    case hipEventInterprocess:
      return "hipEventInterprocess";
    default:
      return "flag combination";
  }
}

static void RunBenchmark(unsigned flag) {
  HipEventCreateWithFlagsBenchmark benchmark;
  benchmark.AddSectionName(GetEventCreateFlagName(flag));
  benchmark.Run(flag);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipEventCreateWithFlags` with all flags:
 *    -# Flags
 *      - hipEventDefault
 *      - hipEventBlockingSync
 *      - hipEventDisableTiming
 *      - hipEventInterprocess (currently disabled)
 * Test source
 * ------------------------
 *  - performance/event/hipEventCreate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipEventCreateWithFlags") {
  const auto flag = GENERATE(
      hipEventDefault, hipEventBlockingSync,
      hipEventDisableTiming /*, hipEventInterprocess  disabled until fixed (EXSWHTEC-25) */);
  RunBenchmark(flag);
}

/**
 * End doxygen group PerformanceTest.
 * @}
 */
