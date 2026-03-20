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

class MemcpyToSymbolAsyncBenchmark : public Benchmark<MemcpyToSymbolAsyncBenchmark> {
 public:
  void operator()(const void* source, size_t size, size_t offset, const hipStream_t& stream) {
    TIMED_SECTION_STREAM(kTimerTypeEvent, stream) {
      HIP_CHECK(hipMemcpyToSymbolAsync(HIP_SYMBOL(devSymbol), source, size, offset,
                                       hipMemcpyHostToDevice, stream));
    }

    HIP_CHECK(hipStreamSynchronize(stream));
  }
};

static void RunBenchmark(const void* source, size_t size = 1, size_t offset = 0) {
  MemcpyToSymbolAsyncBenchmark benchmark;
  benchmark.AddSectionName(std::to_string(size));
  benchmark.AddSectionName(std::to_string(offset));

  const StreamGuard stream_guard(Streams::created);
  const hipStream_t stream = stream_guard.stream();
  benchmark.Run(source, size, offset, stream);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyToSymbolAsync` from Host to Device.
 *  - Utilizes sigular integer values.
 * Test source
 * ------------------------
 *  - performance/memcpy/hipMemcpyToSymbolAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipMemcpyToSymbolAsync_SingularValue) {
  int set{42};
  RunBenchmark(&set);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyToSymbolAsync` from Host to Device.
 *  - Utilizes array integers:
 *    - Small: 1 KB
 *    - Medium: 4 KB
 *    - Large: 1 MB
 * Test source
 * ------------------------
 *  - performance/memcpy/hipMemcpyToSymbolAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipMemcpyToSymbolAsync_ArrayValue) {
  size_t size = GENERATE(1_KB, 4_KB, 1_MB);
  std::vector<int> array(size);
  std::fill_n(array.data(), size, 42);

  RunBenchmark(array.data(), sizeof(int) * size);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyToSymbolAsync` from Host to Device.
 *  - Utilizes array integers with offsets:
 *    - Small: 1 KB
 *    - Medium: 4 KB
 *    - Large: 1 MB
 *  - Offset: 0 and size/2
 * Test source
 * ------------------------
 *  - performance/memcpy/hipMemcpyToSymbolAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipMemcpyToSymbolAsync_WithOffset) {
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
