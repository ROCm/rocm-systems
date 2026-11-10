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

class MemcpyDtoDAsyncBenchmark : public Benchmark<MemcpyDtoDAsyncBenchmark> {
 public:
  void operator()(hipDeviceptr_t& dst, const hipDeviceptr_t& src, size_t size,
                  const hipStream_t& stream) {
    // The whole batch is enqueued back-to-back and drained by the single
    // hipStreamSynchronize that CpuTimer issues before it stops the clock, so the
    // reported time covers enqueue plus drain of all kBatchSize copies.
    TIMED_SECTION_STREAM(kTimerTypeCpu, stream) {
      for (size_t i = 0u; i < kBatchSize; ++i) {
        HIP_CHECK(hipMemcpyDtoDAsync(dst, src, size, stream));
      }
    }
  }
};

static void RunBenchmark(size_t size, bool enable_peer_access = false) {
  MemcpyDtoDAsyncBenchmark benchmark;
  benchmark.AddSectionName(std::to_string(size));
  // A device-to-device copy reads `size` bytes and writes `size` bytes, so the
  // device memory traffic is twice the logical copy size, and the measured time
  // covers the whole batch.
  benchmark.RegisterBandwidth(2 * size * kBatchSize);

  const StreamGuard stream_guard(Streams::created);
  const hipStream_t stream = stream_guard.stream();
  int src_device = std::get<0>(GetDeviceIds(enable_peer_access));
  int dst_device = std::get<1>(GetDeviceIds(enable_peer_access));

  LinearAllocGuard<int> src_allocation(LinearAllocs::hipMalloc, size);
  HIP_CHECK(hipSetDevice(dst_device));
  LinearAllocGuard<int> dst_allocation(LinearAllocs::hipMalloc, size);
  HIP_CHECK(hipSetDevice(src_device));
  benchmark.Run(reinterpret_cast<hipDeviceptr_t>(dst_allocation.ptr()),
                reinterpret_cast<hipDeviceptr_t>(src_allocation.ptr()), size, stream);
}

/**
 * Test Description
 * ------------------------
 *  - Executes a batch of 100 back-to-back `hipMemcpyDtoDAsync` calls from Device to
 *    Device with peer access enabled, enqueued inside a single CPU timed section and
 *    drained by one `hipStreamSynchronize`:
 *    -# Allocation size
 *      - Small: 4 KB, 8 KB, 16 KB
 *      - Medium: 1 MB, 4 MB, 16 MB
 *      - Large: 256 MB
 *    -# Allocation type
 *      - Source: device malloc
 *      - Destination: device malloc
 * Test source
 * ------------------------
 * - performance/api/memcpy/hipMemcpyDtoDAsync.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - Device supports Peer-to-Peer access
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipMemcpyDtoDAsync_PeerAccessEnabled) {
  if (HipTest::getDeviceCount() < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }
  const auto allocation_size = GENERATE(4_KB, 8_KB, 16_KB, 1_MB, 4_MB, 16_MB, 256_MB);
  RunBenchmark(allocation_size, true);
}

/**
 * Test Description
 * ------------------------
 *  - Executes a batch of 100 back-to-back `hipMemcpyDtoDAsync` calls from Device to
 *    Device with peer access disabled, enqueued inside a single CPU timed section and
 *    drained by one `hipStreamSynchronize`:
 *    -# Allocation size
 *      - Small: 4 KB, 8 KB, 16 KB
 *      - Medium: 1 MB, 4 MB, 16 MB
 *      - Large: 256 MB
 *    -# Allocation type
 *      - Source: device malloc
 *      - Destination: device malloc
 * Test source
 * ------------------------
 * - performance/api/memcpy/hipMemcpyDtoDAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipMemcpyDtoDAsync_PeerAccessDisabled) {
  const auto allocation_size = GENERATE(4_KB, 8_KB, 16_KB, 1_MB, 4_MB, 16_MB, 256_MB);
  RunBenchmark(allocation_size);
}

/**
 * End doxygen group memcpy.
 * @}
 */
