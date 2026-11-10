/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "memcpy_performance_common.hh"

/**
 * @addtogroup memcpy memcpy
 * @{
 * @ingroup PerformanceTestMemory
 */

class MemcpyDtoHBenchmark : public Benchmark<MemcpyDtoHBenchmark> {
 public:
  void operator()(void* dst, const hipDeviceptr_t& src, size_t size) {
    TIMED_SECTION(kTimerTypeCpu) { HIP_CHECK(hipMemcpyDtoH(dst, src, size)); }
  }
};

static void RunBenchmark(LinearAllocs host_allocation_type, LinearAllocs device_allocation_type,
                         size_t size) {
  MemcpyDtoHBenchmark benchmark;
  benchmark.AddSectionName(std::to_string(size));
  benchmark.AddSectionName(GetAllocationSectionName(host_allocation_type));
  benchmark.RegisterBandwidth(size);

  LinearAllocGuard<int> device_allocation(device_allocation_type, size);
  LinearAllocGuard<int> host_allocation(host_allocation_type, size);
  benchmark.Run(host_allocation.ptr(), reinterpret_cast<hipDeviceptr_t>(device_allocation.ptr()),
                size);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyDtoH` from Device to Host:
 *    -# Allocation size
 *      - Small: 4 KB, 8 KB, 16 KB
 *      - Medium: 1 MB, 4 MB, 16 MB
 *      - Large: 256 MB
 *    -# Allocation type
 *      - Source: device malloc
 *      - Destination: host pinned and pageable
 * Test source
 * ------------------------
 * - performance/api/memcpy/hipMemcpyDtoH.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipMemcpyDtoH) {
  const auto allocation_size = GENERATE(4_KB, 8_KB, 16_KB, 1_MB, 4_MB, 16_MB, 256_MB);
  const auto device_allocation_type = LinearAllocs::hipMalloc;
  const auto host_allocation_type = GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc);
  RunBenchmark(host_allocation_type, device_allocation_type, allocation_size);
}

/**
 * End doxygen group memcpy.
 * @}
 */
