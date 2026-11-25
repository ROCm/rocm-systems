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

static void RunBenchmark(LinearAllocs dst_allocation_type, LinearAllocs src_allocation_type,
                         size_t size, hipMemcpyKind kind, bool enable_peer_access = false) {
  MemcpyAsyncBenchmark benchmark;
  benchmark.AddSectionName(std::to_string(size));
  benchmark.AddSectionName(GetAllocationSectionName(src_allocation_type));
  benchmark.AddSectionName(GetAllocationSectionName(dst_allocation_type));

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

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyAsync` from Device to Host:
 *    -# Allocation size
 *      - Small: 4 KB
 *      - Medium: 4 MB
 *      - Large: 16 MB
 *    -# Allocation type
 *      - Source: device malloc
 *      - Destination: host pinned and pageable
 * Test source
 * ------------------------
 * - performance/api/memcpy/hipMemcpyAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipMemcpyAsync_DeviceToHost) {
  const auto allocation_size = GENERATE(4_KB, 4_MB, 16_MB);
  const auto src_allocation_type = LinearAllocs::hipMalloc;
  const auto dst_allocation_type = GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc);
  RunBenchmark(dst_allocation_type, src_allocation_type, allocation_size, hipMemcpyDeviceToHost);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyAsync` from Host to Device:
 *    -# Allocation size
 *      - Small: 4 KB
 *      - Medium: 4 MB
 *      - Large: 16 MB
 *    -# Allocation type
 *      - Source: host pinned and pageable
 *      - Destination: device malloc
 * Test source
 * ------------------------
 * - performance/api/memcpy/hipMemcpyAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipMemcpyAsync_HostToDevice) {
  const auto allocation_size = GENERATE(4_KB, 4_MB, 16_MB);
  const auto src_allocation_type = GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc);
  const auto dst_allocation_type = LinearAllocs::hipMalloc;
  RunBenchmark(dst_allocation_type, src_allocation_type, allocation_size, hipMemcpyHostToDevice);
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
 *      - Larger than the last-level cache: 512 MB
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
  const auto allocation_size = GENERATE(4_KB, 4_MB, 16_MB, 512_MB);
  size_t free_memory = 0, total_memory = 0;
  HIP_CHECK(hipMemGetInfo(&free_memory, &total_memory));
  if (free_memory < 2 * allocation_size + 1_MB) {
    HIP_SKIP_TEST("Not enough free device memory");
  }
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

// Times on the host what a caller waits for: the copy's submission, the copy, and the stream
// synchronize, which the CPU timer runs when the section ends. An event timer sees only the copy.
class MemcpyAsyncSynchronizedBenchmark : public Benchmark<MemcpyAsyncSynchronizedBenchmark> {
 public:
  void operator()(void* dst, const void* src, size_t size, hipMemcpyKind kind,
                  const hipStream_t& stream) {
    TIMED_SECTION_STREAM(kTimerTypeCpu, stream) {
      HIP_CHECK(hipMemcpyAsync(dst, src, size, kind, stream));
    }
  }
};

__global__ void MemcpyAsyncDependentKernel() {}

// The same, with a kernel that waits for the copy before the stream synchronize
class MemcpyAsyncDependentKernelBenchmark
    : public Benchmark<MemcpyAsyncDependentKernelBenchmark> {
 public:
  void operator()(void* dst, const void* src, size_t size, const hipStream_t& stream) {
    TIMED_SECTION_STREAM(kTimerTypeCpu, stream) {
      HIP_CHECK(hipMemcpyAsync(dst, src, size, hipMemcpyHostToDevice, stream));
      MemcpyAsyncDependentKernel<<<1, 1, 0, stream>>>();
      HIP_CHECK(hipGetLastError());
    }
  }
};

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyAsync` between pinned host memory and device memory and synchronizes the
 *    stream, timed on the host, so the time includes submitting the copy and noticing that it
 *    completed:
 *    -# Allocation size
 *      - 256 KB, 1 MB, 4 MB
 *    -# Direction
 *      - Host to Device and Device to Host
 * Test source
 * ------------------------
 * - performance/api/memcpy/hipMemcpyAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Performance_hipMemcpyAsync_PinnedSynchronized) {
  const auto allocation_size = GENERATE(256_KB, 1_MB, 4_MB);
  const auto kind = GENERATE(hipMemcpyHostToDevice, hipMemcpyDeviceToHost);
  MemcpyAsyncSynchronizedBenchmark benchmark;
  benchmark.AddSectionName(std::to_string(allocation_size));
  benchmark.AddSectionName(kind == hipMemcpyHostToDevice ? "host to device" : "device to host");
  benchmark.RegisterBandwidth(allocation_size);

  const StreamGuard stream_guard{Streams::created};
  LinearAllocGuard<int> host_allocation(LinearAllocs::hipHostMalloc, allocation_size);
  LinearAllocGuard<int> device_allocation(LinearAllocs::hipMalloc, allocation_size);
  if (kind == hipMemcpyHostToDevice) {
    benchmark.Run(device_allocation.ptr(), host_allocation.ptr(), allocation_size, kind,
                  stream_guard.stream());
  } else {
    benchmark.Run(host_allocation.ptr(), device_allocation.ptr(), allocation_size, kind,
                  stream_guard.stream());
  }
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyAsync` from pinned host memory to device memory, then a kernel on the same
 *    stream that waits for the copy, and synchronizes the stream, timed on the host:
 *    -# Allocation size
 *      - 256 KB, 1 MB
 * Test source
 * ------------------------
 * - performance/api/memcpy/hipMemcpyAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Performance_hipMemcpyAsync_HostToDevice_DependentKernel) {
  const auto allocation_size = GENERATE(256_KB, 1_MB);
  MemcpyAsyncDependentKernelBenchmark benchmark;
  benchmark.AddSectionName(std::to_string(allocation_size));

  const StreamGuard stream_guard{Streams::created};
  LinearAllocGuard<int> host_allocation(LinearAllocs::hipHostMalloc, allocation_size);
  LinearAllocGuard<int> device_allocation(LinearAllocs::hipMalloc, allocation_size);
  benchmark.Run(device_allocation.ptr(), host_allocation.ptr(), allocation_size,
                stream_guard.stream());
}

/**
 * End doxygen group memcpy.
 * @}
 */
