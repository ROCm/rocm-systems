/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "memcpy_performance_common.hh"

#include <cstdlib>

/**
 * @addtogroup memcpy memcpy
 * @{
 * @ingroup PerformanceTestMemory
 */

class MemcpyAsyncBenchmark : public Benchmark<MemcpyAsyncBenchmark> {
 public:
  void operator()(void* dst, const void* src, size_t size, hipMemcpyKind kind,
                  const hipStream_t& stream) {
    TIMED_SECTION_STREAM(kTimerTypeEvent, stream) {
      HIP_CHECK(hipMemcpyAsync(dst, src, size, kind, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
  }
};

static constexpr size_t kCopySizes[] = {
    1_KB, 2_KB, 4_KB, 8_KB, 16_KB, 32_KB, 64_KB, 128_KB, 256_KB, 512_KB,
    1_MB, 2_MB, 4_MB, 8_MB, 16_MB};

static void RunBenchmark(LinearAllocs dst_allocation_type, LinearAllocs src_allocation_type,
                         size_t size, hipMemcpyKind kind, bool enable_peer_access = false) {
  MemcpyAsyncBenchmark benchmark;
  benchmark.AddSectionName(std::to_string(size));
  benchmark.AddSectionName(GetAllocationSectionName(src_allocation_type));
  benchmark.AddSectionName(GetAllocationSectionName(dst_allocation_type));
  benchmark.RegisterBandwidth(size);

  const StreamGuard stream_guard{Streams::created};
  const hipStream_t stream = stream_guard.stream();
  if (kind != hipMemcpyDeviceToDevice) {
    LinearAllocGuard<int> src_allocation(src_allocation_type, size);
    LinearAllocGuard<int> dst_allocation(dst_allocation_type, size);
    benchmark.Run(dst_allocation.ptr(), src_allocation.ptr(), size, kind, stream);
  } else {
    int src_device = std::get<0>(GetDeviceIds(enable_peer_access));
    int dst_device = std::get<1>(GetDeviceIds(enable_peer_access));

    LinearAllocGuard<int> src_allocation(src_allocation_type, size);
    HIP_CHECK(hipSetDevice(dst_device));
    LinearAllocGuard<int> dst_allocation(dst_allocation_type, size);
    HIP_CHECK(hipSetDevice(src_device));
    benchmark.Run(dst_allocation.ptr(), src_allocation.ptr(), size, kind, stream);
  }
}

static void RunPageableCopySizes(LinearAllocs dst_allocation_type, LinearAllocs src_allocation_type,
                                 hipMemcpyKind kind) {
  const char* env = std::getenv("HIP_BENCH_COPY_SIZE");
  if (env != nullptr) {
    RunBenchmark(dst_allocation_type, src_allocation_type, std::stoull(env), kind);
    return;
  }
  for (size_t size : kCopySizes) {
    RunBenchmark(dst_allocation_type, src_allocation_type, size, kind);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyAsync` from Device to Host:
 *    -# Allocation size: 1 KB through 16 MB
 *    -# Allocation type
 *      - Source: device malloc
 *      - Destination: host pageable (malloc)
 * Test source
 * ------------------------
 * - performance/api/memcpy/hipMemcpyAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipMemcpyAsync_DeviceToHost) {
  const auto src_allocation_type = LinearAllocs::hipMalloc;
  const auto dst_allocation_type = LinearAllocs::malloc;
  RunPageableCopySizes(dst_allocation_type, src_allocation_type, hipMemcpyDeviceToHost);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyAsync` from Host to Device:
 *    -# Allocation size: 1 KB through 16 MB
 *    -# Allocation type
 *      - Source: host pageable (malloc)
 *      - Destination: device malloc
 * Test source
 * ------------------------
 * - performance/api/memcpy/hipMemcpyAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipMemcpyAsync_HostToDevice) {
  const auto src_allocation_type = LinearAllocs::malloc;
  const auto dst_allocation_type = LinearAllocs::hipMalloc;
  RunPageableCopySizes(dst_allocation_type, src_allocation_type, hipMemcpyHostToDevice);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyAsync` from Host to Host:
 *    -# Allocation size
 *      - Small: 4 KB
 *      - Medium: 4 MB
 *      - Large: 16 MB
 *    -# Allocation type
 *      - Source: host pinned and pageable
 *      - Destination: host pinned and pageable
 * Test source
 * ------------------------
 * - performance/api/memcpy/hipMemcpyAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipMemcpyAsync_HostToHost) {
  const auto allocation_size = GENERATE(4_KB, 4_MB, 16_MB);
  const auto src_allocation_type = GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc);
  const auto dst_allocation_type = GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc);
  RunBenchmark(dst_allocation_type, src_allocation_type, allocation_size, hipMemcpyHostToHost);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyAsync` from Device to Device with peer access disabled:
 *    -# Allocation size
 *      - Small: 4 KB
 *      - Medium: 4 MB
 *      - Large: 16 MB
 *    -# Allocation type
 *      - Source: device malloc
 *      - Destination: device malloc
 * Test source
 * ------------------------
 * - performance/api/memcpy/hipMemcpyAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipMemcpyAsync_DeviceToDevice_DisablePeerAccess) {
  const auto allocation_size = GENERATE(4_KB, 4_MB, 16_MB);
  const auto src_allocation_type = LinearAllocs::hipMalloc;
  const auto dst_allocation_type = LinearAllocs::hipMalloc;
  RunBenchmark(dst_allocation_type, src_allocation_type, allocation_size, hipMemcpyDeviceToDevice);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyAsync` from Device to Device with peer access enabled:
 *    -# Allocation size
 *      - Small: 4 KB
 *      - Medium: 4 MB
 *      - Large: 16 MB
 *    -# Allocation type
 *      - Source: device malloc
 *      - Destination: device malloc
 * Test source
 * ------------------------
 * - performance/api/memcpy/hipMemcpyAsync.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - Device supports Peer-to-Peer access
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipMemcpyAsync_DeviceToDevice_EnablePeerAccess) {
  if (HipTest::getDeviceCount() < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }
  const auto allocation_size = GENERATE(4_KB, 4_MB, 16_MB);
  const auto src_allocation_type = LinearAllocs::hipMalloc;
  const auto dst_allocation_type = LinearAllocs::hipMalloc;
  RunBenchmark(dst_allocation_type, src_allocation_type, allocation_size, hipMemcpyDeviceToDevice,
               true);
}

/**
 * End doxygen group memcpy.
 * @}
 */
