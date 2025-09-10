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
 * @addtogroup stream stream
 * @{
 * @ingroup PerformanceTest
 * Contains performance tests for all hipStream related APIs
 */

class HipDeviceGetStreamPriorityRangeBenchmark
    : public Benchmark<HipDeviceGetStreamPriorityRangeBenchmark> {
 public:
  void operator()() {
    int priority_min, priority_max;
    TIMED_SECTION(kTimerTypeCpu) {
      HIP_CHECK(hipDeviceGetStreamPriorityRange(&priority_min, &priority_max));
    }
  }
};

class HipStreamQueryBenchmark : public Benchmark<HipStreamQueryBenchmark> {
 public:
  void operator()(bool perform_work) {
    hipError_t error;
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    void* dptr;

    if (perform_work) {
      HIP_CHECK(hipMallocAsync(&dptr, 2048 * 4, stream));
    }

    TIMED_SECTION(kTimerTypeCpu) { error = hipStreamQuery(stream); }

    if (perform_work) {
      HIP_CHECK(hipFreeAsync(dptr, stream));
      HIP_CHECK(hipStreamSynchronize(stream));
    }

    HIP_CHECK(hipStreamDestroy(stream));
  }
};

class HipStreamSynchronizeBenchmark : public Benchmark<HipStreamSynchronizeBenchmark> {
 public:
  void operator()() {
    hipError_t error;
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    TIMED_SECTION(kTimerTypeCpu) { error = hipStreamSynchronize(stream); }

    HIP_CHECK(hipStreamDestroy(stream));
  }
};

class HipStreamDestroyBenchmark : public Benchmark<HipStreamDestroyBenchmark> {
 public:
  void operator()() {
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    TIMED_SECTION(kTimerTypeCpu) { HIP_CHECK(hipStreamDestroy(stream)); }
  }
};

class HipStreamCreateBenchmark : public Benchmark<HipStreamCreateBenchmark> {
 public:
  void operator()() {
    hipStream_t stream;

    TIMED_SECTION(kTimerTypeCpu) { HIP_CHECK(hipStreamCreate(&stream)); }

    HIP_CHECK(hipStreamDestroy(stream));
  }
};

class HipStreamCreateWithPriorityBenchmark
    : public Benchmark<HipStreamCreateWithPriorityBenchmark> {
 public:
  void operator()(unsigned int flag) {
    hipStream_t stream;
    int priority_min, priority_max, priority_mid;

    HIP_CHECK(hipDeviceGetStreamPriorityRange(&priority_min, &priority_max));
    priority_mid = (priority_max + priority_min) / 2;

    TIMED_SECTION(kTimerTypeCpu) {
      HIP_CHECK(hipStreamCreateWithPriority(&stream, flag, priority_mid));
    }

    HIP_CHECK(hipStreamDestroy(stream));
  }
};


static std::string GetStreamCreateFlagName(unsigned flag) {
  switch (flag) {
    case hipStreamDefault:
      return "hipStreamDefault";
    case hipStreamNonBlocking:
      return "hipStreamNonBlocking";
    default:
      return "flag combination";
  }
}

class HipStreamCreateWithFlagsBenchmark : public Benchmark<HipStreamCreateWithFlagsBenchmark> {
 public:
  void operator()(unsigned int flag) {
    hipStream_t stream;

    TIMED_SECTION(kTimerTypeCpu) { HIP_CHECK(hipStreamCreateWithFlags(&stream, flag)); }

    HIP_CHECK(hipStreamDestroy(stream));
  }
};

/**
 * Test Description
 * ------------------------
 *  - Executes `hipStreamCreate`:
 * Test source
 * ------------------------
 *  - performance/stream/hipStreamBasic.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipStreamCreate") {
  HipStreamCreateBenchmark benchmark;
  benchmark.Run();
}

static void RunBenchmark(unsigned flag) {
  HipStreamCreateWithFlagsBenchmark benchmark;
  benchmark.AddSectionName(GetStreamCreateFlagName(flag));
  benchmark.Run(flag);
}

static void RunBenchmarkWithPriority(unsigned flag) {
  HipStreamCreateWithPriorityBenchmark benchmark;
  benchmark.AddSectionName(GetStreamCreateFlagName(flag));
  benchmark.Run(flag);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipStreamCreateWithFlags` with all flags:
 *    -# Flags
 *      - hipStreamDefault
 *      - hipStreamNonBlocking
 * Test source
 * ------------------------
 *  - performance/stream/hipStreamBasic.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipStreamCreateWithFlags") {
  const auto flag = GENERATE(hipStreamDefault, hipStreamNonBlocking);
  RunBenchmark(flag);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipStreamCreateWithPriority` with all flags:
 *    -# Flags
 *      - hipStreamDefault
 *      - hipStreamNonBlocking
 * Test source
 * ------------------------
 *  - performance/stream/hipStreamBasic.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipStreamCreateWithPriority") {
  const auto flag = GENERATE(hipStreamDefault, hipStreamNonBlocking);
  RunBenchmarkWithPriority(flag);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipStreamDestroy`:
 * Test source
 * ------------------------
 *  - performance/stream/hipStreamBasic.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipStreamDestroy") {
  HipStreamDestroyBenchmark benchmark;
  benchmark.Run();
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipDeviceGetStreamPriorityRange`:
 * Test source
 * ------------------------
 *  - performance/stream/hipStreamBasic.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipDeviceGetStreamPriorityRange") {
  HipDeviceGetStreamPriorityRangeBenchmark benchmark;
  benchmark.Run();
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipStreamQuery`:
 * Test source
 * ------------------------
 *  - performance/stream/hipStreamBasic.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipStreamQuery") {
  const auto perform_work = GENERATE(true, false);
  HipStreamQueryBenchmark benchmark;
  if (perform_work) {
    benchmark.AddSectionName("stream with work");
  } else {
    benchmark.AddSectionName("stream without work");
  }
  benchmark.Run(perform_work);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipDeviceGetStreamPriorityRange`:
 * Test source
 * ------------------------
 *  - performance/stream/hipStreamBasic.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipStreamSynchronize") {
  HipStreamSynchronizeBenchmark benchmark;
  benchmark.Run();
}

/**
 * End doxygen group PerformanceTest.
 * @}
 */
