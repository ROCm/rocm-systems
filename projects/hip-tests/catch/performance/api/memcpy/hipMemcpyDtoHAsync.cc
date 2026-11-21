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

// Number of back-to-back async copies enqueued per measured iteration.
constexpr size_t kBatchSize = 100;

class MemcpyDtoHAsyncBenchmark : public Benchmark<MemcpyDtoHAsyncBenchmark> {
 public:
  void operator()(void* dst, const hipDeviceptr_t& src, size_t size, const hipStream_t& stream) {
    // The whole batch is enqueued back-to-back and drained by the single
    // hipStreamSynchronize that CpuTimer issues before it stops the clock, so the
    // reported time covers enqueue plus drain of all kBatchSize copies.
    TIMED_SECTION_STREAM(kTimerTypeCpu, stream) {
      for (size_t i = 0u; i < kBatchSize; ++i) {
        HIP_CHECK(hipMemcpyDtoHAsync(dst, src, size, stream));
      }
    }
  }
};

static void RunBenchmark(LinearAllocs host_allocation_type, LinearAllocs device_allocation_type,
                         size_t size) {
  MemcpyDtoHAsyncBenchmark benchmark;
  benchmark.AddSectionName(std::to_string(size));
  benchmark.AddSectionName(GetAllocationSectionName(host_allocation_type));
  // The measured time covers the whole batch, so the transferred bytes must too.
  benchmark.RegisterBandwidth(size * kBatchSize);

  const StreamGuard stream_guard(Streams::created);
  const hipStream_t stream = stream_guard.stream();
  LinearAllocGuard<int> device_allocation(device_allocation_type, size);
  LinearAllocGuard<int> host_allocation(host_allocation_type, size);
  benchmark.Run(host_allocation.ptr(), reinterpret_cast<hipDeviceptr_t>(device_allocation.ptr()),
                size, stream);
}

/**
 * Test Description
 * ------------------------
 *  - Executes a batch of 100 back-to-back `hipMemcpyDtoHAsync` calls from Device to
 *    Host, enqueued inside a single CPU timed section and drained by one
 *    `hipStreamSynchronize`:
 *    -# Allocation size
 *      - Small: 4 KB, 8 KB, 16 KB
 *      - Medium: 1 MB, 4 MB, 16 MB
 *      - Large: 256 MB
 *    -# Allocation type
 *      - Source: device malloc
 *      - Destination: host pinned and pageable
 * Test source
 * ------------------------
 * - performance/api/memcpy/hipMemcpyDtoHAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipMemcpyDtoHAsync) {
  const auto allocation_size = GENERATE(4_KB, 8_KB, 16_KB, 1_MB, 4_MB, 16_MB, 256_MB);
  const auto device_allocation_type = LinearAllocs::hipMalloc;
  const auto host_allocation_type = GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc);
  RunBenchmark(host_allocation_type, device_allocation_type, allocation_size);
}

/**
 * End doxygen group memcpy.
 * @}
 */
