/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <performance_common.hh>
#include <resource_guards.hh>

/**
 * @addtogroup memset memset
 * @{
 * @ingroup PerformanceTest
 * Contains performance tests for all memset HIP APIs.
 */

class MemsetBenchmark : public Benchmark<MemsetBenchmark> {
 public:
  MemsetBenchmark(LinearAllocs allocation_type, size_t size)
      : dst_(allocation_type, size), size_(size) {}

  void operator()() {
    TIMED_SECTION(kTimerTypeEvent) { HIP_CHECK(hipMemset(dst_.ptr(), 17, size_)); }
  }

 private:
  LinearAllocGuard<void> dst_;
  const size_t size_;
};

static void RunBenchmark(LinearAllocs allocation_type, size_t size) {
  MemsetBenchmark benchmark(allocation_type, size);
  benchmark.AddSectionName(std::to_string(size));
  benchmark.AddSectionName(GetAllocationSectionName(allocation_type));
  benchmark.Run();
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemset`:
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
 *  - performance/memset/hipMemset.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Performance_hipMemset) {
  const auto size = GENERATE(4_KB, 4_MB, 16_MB);
  const auto allocation_type = GENERATE(LinearAllocs::hipMalloc, LinearAllocs::hipHostMalloc,
                                        LinearAllocs::hipMallocManaged);
  RunBenchmark(allocation_type, size);
}

/**
 * End doxygen group PerformanceTest.
 * @}
 */
