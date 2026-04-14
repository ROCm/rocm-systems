/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "memcpy_performance_common.hh"
#pragma clang diagnostic ignored "-Wvla-extension"

/**
 * @addtogroup memcpy memcpy
 * @{
 * @ingroup PerformanceTest
 */

__device__ int devSymbol[1_MB];

class MemcpyFromSymbolAsyncBenchmark : public Benchmark<MemcpyFromSymbolAsyncBenchmark> {
 public:
  void operator()(const void* source, void* result, size_t size, size_t offset,
                  const hipStream_t& stream) {
    HIP_CHECK(hipMemcpyToSymbolAsync(HIP_SYMBOL(devSymbol), source, size, offset,
                                     hipMemcpyHostToDevice, stream));
    TIMED_SECTION_STREAM(kTimerTypeEvent, stream) {
      HIP_CHECK(hipMemcpyFromSymbolAsync(result, HIP_SYMBOL(devSymbol), size, offset,
                                         hipMemcpyDeviceToHost, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
  }
};

static void RunBenchmark(const void* source, void* result, size_t size = 1, size_t offset = 0) {
  MemcpyFromSymbolAsyncBenchmark benchmark;
  benchmark.AddSectionName(std::to_string(size));
  benchmark.AddSectionName(std::to_string(offset));

  const StreamGuard stream_guard(Streams::created);
  const hipStream_t stream = stream_guard.stream();
  benchmark.Run(source, result, size, offset, stream);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyFromSymbolAsync` from Device to Host.
 *  - Utilizes sigular integer values.
 * Test source
 * ------------------------
 *  - performance/memcpy/hipMemcpyFromSymbolAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Performance_hipMemcpyFromSymbolAsync_SingularValue) {
  int set{42};
  int result{0};
  RunBenchmark(&set, &result);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyFromSymbolAsync` from Device to Host.
 *  - Utilizes array integers:
 *    - Small: 1 KB
 *    - Medium: 4 KB
 *    - Large: 512 KB
 * Test source
 * ------------------------
 *  - performance/memcpy/hipMemcpyFromSymbolAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Performance_hipMemcpyFromSymbolAsync_ArrayValue) {
  size_t size = GENERATE(1_KB, 4_KB, 512_KB);
  std::vector<int> array(size);
  std::fill_n(array.data(), size, 42);
  std::vector<int> result(size);
  std::fill_n(result.data(), size, 0);

  RunBenchmark(array.data(), result.data(), sizeof(int) * size);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyFromSymbolAsync` from Device to Host.
 *  - Utilizes array integers with offsets:
 *    - Small: 1 KB
 *    - Medium: 4 KB
 *    - Large: 512 KB
 *  - Offset: 0 and size/2
 * Test source
 * ------------------------
 *  - performance/memcpy/hipMemcpyFromSymbolAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Performance_hipMemcpyFromSymbolAsync_WithOffset) {
  size_t size = GENERATE(1_KB, 4_KB, 512_KB);
  std::vector<int> array(size);
  std::fill_n(array.data(), size, 42);
  std::vector<int> result(size);
  std::fill_n(result.data(), size, 0);

  size_t offset = GENERATE_REF(0, size / 2);
  RunBenchmark(array.data() + offset, result.data() + offset, sizeof(int) * (size - offset),
               offset * sizeof(int));
}

/**
 * End doxygen group memcpy.
 * @}
 */
