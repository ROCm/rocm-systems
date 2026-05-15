/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "memcpy_performance_common.hh"
#pragma clang diagnostic ignored "-Wvla-extension"
/**
 * @addtogroup memcpy memcpy
 * @{
 * @ingroup PerformanceTest
 */

__device__ int devSymbol[1_MB];

class MemcpyToSymbolBenchmark : public Benchmark<MemcpyToSymbolBenchmark> {
 public:
  void operator()(const void* source, size_t size, size_t offset) {
    TIMED_SECTION(kTimerTypeCpu) {
      HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(devSymbol), source, size, offset));
    }
  }
};

static void RunBenchmark(const void* source, size_t size = 1, size_t offset = 0) {
  MemcpyToSymbolBenchmark benchmark;
  benchmark.AddSectionName(std::to_string(size));
  benchmark.AddSectionName(std::to_string(offset));
  benchmark.Run(source, size, offset);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyToSymbol` from Host to Device.
 *  - Utilizes sigular integer values.
 * Test source
 * ------------------------
 *  - performance/memcpy/hipMemcpyToSymbol.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpyToSymbol_SingularValue") {
  int set{42};
  RunBenchmark(&set);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyToSymbol` from Host to Device.
 *  - Utilizes array integers:
 *    - Small: 1 KB
 *    - Medium: 4 KB
 *    - Large: 1 MB
 * Test source
 * ------------------------
 *  - performance/memcpy/hipMemcpyToSymbol.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpyToSymbol_ArrayValue") {
  size_t size = GENERATE(1_KB, 4_KB, 1_MB);
  std::vector<int> array(size);
  std::fill_n(array.data(), size, 42);

  RunBenchmark(array.data(), sizeof(int) * size);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyToSymbol` from Host to Device.
 *  - Utilizes array integers with offsets:
 *    - Small: 1 KB
 *    - Medium: 4 KB
 *    - Large: 1 MB
 *  - Offset: 0 and size/2
 * Test source
 * ------------------------
 *  - performance/memcpy/hipMemcpyToSymbol.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpyToSymbol_WithOffset") {
  size_t size = GENERATE(1_KB, 4_KB, 1_MB);
  std::vector<int> array(size);
  std::fill_n(array.data(), size, 42);

  size_t offset = GENERATE_REF(0, size / 2);
  RunBenchmark(array.data() + offset, sizeof(int) * (size - offset), offset * sizeof(int));
}

/**
 * End doxygen group memcpy.
 * @}
 */
