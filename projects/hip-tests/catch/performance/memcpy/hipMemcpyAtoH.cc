/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "memcpy_performance_common.hh"

/**
 * @addtogroup memcpy memcpy
 * @{
 * @ingroup PerformanceTest
 */

class MemcpyAtoHBenchmark : public Benchmark<MemcpyAtoHBenchmark> {
 public:
  void operator()(void* dst, hipArray_t src_array, size_t allocation_size) {
    TIMED_SECTION(kTimerTypeCpu) { HIP_CHECK(hipMemcpyAtoH(dst, src_array, 0, allocation_size)); }
  }
};

static void RunBenchmark(LinearAllocs host_allocation_type, size_t width) {
  MemcpyAtoHBenchmark benchmark;
  benchmark.AddSectionName(std::to_string(width));
  benchmark.AddSectionName(GetAllocationSectionName(host_allocation_type));

  size_t allocation_size = width * sizeof(int);
  LinearAllocGuard<int> host_allocation(host_allocation_type, allocation_size);
  ArrayAllocGuard<int> array_allocation(make_hipExtent(width, 0, 0), hipArrayDefault);
  benchmark.Run(host_allocation.ptr(), array_allocation.ptr(), allocation_size);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyAtoH` from Device array to Host:
 *    -# Allocation size
 *      - Small: 512 B
 *      - Medium: 1024 B
 *      - Large: 4096 B
 *    -# Allocation type
 *      - Host: host pinned and pageable
 * Test source
 * ------------------------
 *  - performance/memcpy/hipMemcpyAtoH.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Performance_hipMemcpyAtoH) {
  CHECK_IMAGE_SUPPORT

  const auto allocation_size = GENERATE(512, 1024, 4096);
  const auto host_allocation_type = GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc);
  RunBenchmark(host_allocation_type, allocation_size);
}

/**
 * End doxygen group memcpy.
 * @}
 */
