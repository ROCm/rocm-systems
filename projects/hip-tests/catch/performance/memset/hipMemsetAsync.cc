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

class MemsetAsyncBenchmark : public Benchmark<MemsetAsyncBenchmark> {
 public:
  MemsetAsyncBenchmark(LinearAllocs allocation_type, size_t size)
      : dst_(allocation_type, size), size_(size), stream_(Streams::created) {}

  void operator()() {
    TIMED_SECTION_STREAM(kTimerTypeEvent, stream_.stream()) {
      HIP_CHECK(hipMemsetAsync(dst_.ptr(), 17, size_, stream_.stream()));
    }
    HIP_CHECK(hipStreamSynchronize(stream_.stream()));
  }

 private:
  LinearAllocGuard<void> dst_;
  const size_t size_;
  StreamGuard stream_;
};

static void RunBenchmark(LinearAllocs allocation_type, size_t size) {
  MemsetAsyncBenchmark benchmark(allocation_type, size);
  benchmark.AddSectionName(std::to_string(size));
  benchmark.AddSectionName(GetAllocationSectionName(allocation_type));
  benchmark.Run();
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemsetAsync`:
 *    -# Allocation size
 *      - Small: 4 KB
 *      - Medium: 4 MB
 *      - Large: 16 MB
 *    -# Allocation type
 *      - device
 *      - host
 *      - managed
 * Test source
 * ------------------------
 *  - performance/memset/hipMemsetAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemsetAsync") {
  const auto size = GENERATE(4_KB, 4_MB, 16_MB);
  const auto allocation_type = GENERATE(LinearAllocs::hipMalloc, LinearAllocs::hipHostMalloc,
                                        LinearAllocs::hipMallocManaged);
  RunBenchmark(allocation_type, size);
}

/**
 * End doxygen group PerformanceTest.
 * @}
 */
