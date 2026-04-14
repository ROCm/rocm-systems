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
 */

class MemsetD16Benchmark : public Benchmark<MemsetD16Benchmark> {
 public:
  MemsetD16Benchmark(LinearAllocs allocation_type, size_t size)
      : dst_(allocation_type, size * sizeof(int16_t)), size_(size) {}

  void operator()() {
    TIMED_SECTION(kTimerTypeEvent) {
      HIP_CHECK(hipMemsetD16(reinterpret_cast<hipDeviceptr_t>(dst_.ptr()), 311, size_));
    }
  }

 private:
  LinearAllocGuard<void> dst_;
  const size_t size_;
};

static void RunBenchmark(LinearAllocs allocation_type, size_t size) {
  MemsetD16Benchmark benchmark(allocation_type, size);
  benchmark.AddSectionName(std::to_string(size));
  benchmark.AddSectionName(GetAllocationSectionName(allocation_type));
  benchmark.Run();
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemsetD16`:
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
 *  - performance/memset/hipMemsetD16.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Performance_hipMemsetD16) {
  const auto size = GENERATE(4_KB, 4_MB, 16_MB);
  const auto allocation_type = GENERATE(LinearAllocs::hipMalloc, LinearAllocs::hipHostMalloc,
                                        LinearAllocs::hipMallocManaged);
  RunBenchmark(allocation_type, size);
}

/**
 * End doxygen group PerformanceTest.
 * @}
 */
