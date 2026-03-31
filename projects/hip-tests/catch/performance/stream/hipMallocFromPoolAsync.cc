/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "mem_pools_performance_common.hh"

/**
 * @addtogroup stream stream
 * @{
 * @ingroup PerformanceTest
 */

class MallocFromPoolAsyncBenchmark : public Benchmark<MallocFromPoolAsyncBenchmark> {
 public:
  void operator()(const size_t array_size) {
    const StreamGuard stream_guard{Streams::created};
    const hipStream_t stream = stream_guard.stream();

    hipMemPool_t mem_pool{nullptr};
    hipMemPoolProps pool_props = CreateMemPoolProps(0, hipMemHandleTypeNone);
    HIP_CHECK(hipMemPoolCreate(&mem_pool, &pool_props));

    float* array_ptr{nullptr};

    TIMED_SECTION_STREAM(kTimerTypeEvent, stream) {
      HIP_CHECK(hipMallocFromPoolAsync(&array_ptr, array_size * sizeof(float), mem_pool, stream));
    }

    REQUIRE(array_ptr != nullptr);

    HIP_CHECK(hipFreeAsync(array_ptr, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemPoolDestroy(mem_pool));
  }
};

static void RunBenchmark(const size_t array_size) {
  MallocFromPoolAsyncBenchmark benchmark;
  benchmark.AddSectionName(std::to_string(array_size));
  benchmark.Run(array_size);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMallocFromPoolAsync`:
 *    -# Allocation size:
 *      - 4 KB
 *      - 4 MB
 *      - 16 MB
 * Test source
 * ------------------------
 *  - performance/stream/hipMallocFromPoolAsync.cc
 * Test requirements
 * ------------------------
 *  - Device supports memory pools
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipMallocFromPoolAsync) {
  if (!AreMemPoolsSupported(0)) {
    HipTest::HIP_SKIP_TEST(
        "GPU 0 doesn't support hipDeviceAttributeMemoryPoolsSupported "
        "attribute. Hence skipping the testing with Pass result.\n");
    return;
  }
  size_t array_size = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(array_size);
}

/**
 * End doxygen group PerformanceTest.
 * @}
 */
