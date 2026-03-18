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

class MemcpyHtoDBenchmark : public Benchmark<MemcpyHtoDBenchmark> {
 public:
  void operator()(hipDeviceptr_t& dst, void* src, size_t size) {
    TIMED_SECTION(kTimerTypeCpu) { HIP_CHECK(hipMemcpyHtoD(dst, src, size)); }
  }
};

static void RunBenchmark(LinearAllocs host_allocation_type, LinearAllocs device_allocation_type,
                         size_t size) {
  MemcpyHtoDBenchmark benchmark;
  benchmark.AddSectionName(std::to_string(size));
  benchmark.AddSectionName(GetAllocationSectionName(host_allocation_type));

  LinearAllocGuard<int> device_allocation(device_allocation_type, size);
  LinearAllocGuard<int> host_allocation(host_allocation_type, size);
  benchmark.Run(reinterpret_cast<hipDeviceptr_t>(device_allocation.ptr()), host_allocation.ptr(),
                size);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyHtoD` from Host to Device:
 *    -# Allocation size
 *      - Small: 4 KB
 *      - Medium: 4 MB
 *      - Large: 16 MB
 *    -# Allocation type
 *      - Source: host pinned and pageable
 *      - Destination: device malloc
 * Test source
 * ------------------------
 *  - performance/memcpy/hipMemcpyHtoD.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Performance_hipMemcpyHtoD) {
  const auto allocation_size = GENERATE(4_KB, 4_MB, 16_MB);
  const auto device_allocation_type = LinearAllocs::hipMalloc;
  const auto host_allocation_type = GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc);
  RunBenchmark(host_allocation_type, device_allocation_type, allocation_size);
}

/**
 * End doxygen group memcpy.
 * @}
 */
