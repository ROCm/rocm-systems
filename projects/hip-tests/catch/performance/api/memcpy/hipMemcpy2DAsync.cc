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

class Memcpy2DAsyncBenchmark : public Benchmark<Memcpy2DAsyncBenchmark> {
 public:
  void operator()(void* dst, size_t dst_pitch, const void* src, size_t src_pitch, size_t width,
                  size_t height, hipMemcpyKind kind, const hipStream_t& stream) {
    TIMED_SECTION_STREAM(kTimerTypeEvent, stream) {
      HIP_CHECK(hipMemcpy2DAsync(dst, dst_pitch, src, src_pitch, width, height, kind, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
  }
};

static void RunBenchmark(size_t width, size_t height, hipMemcpyKind kind,
                         bool enable_peer_access = false) {
  Memcpy2DAsyncBenchmark benchmark;
  benchmark.AddSectionName("(" + std::to_string(width) + ", " + std::to_string(height) + ")");

  const StreamGuard stream_guard(Streams::created);
  const hipStream_t stream = stream_guard.stream();

  if (kind == hipMemcpyDeviceToHost) {
    LinearAllocGuard2D<int> device_allocation(width, height);
    LinearAllocGuard<int> host_allocation(LinearAllocs::hipHostMalloc,
                                          device_allocation.width() * height);
    benchmark.Run(host_allocation.ptr(), device_allocation.width(), device_allocation.ptr(),
                  device_allocation.pitch(), device_allocation.width(), device_allocation.height(),
                  hipMemcpyDeviceToHost, stream);
  } else if (kind == hipMemcpyHostToDevice) {
    LinearAllocGuard2D<int> device_allocation(width, height);
    LinearAllocGuard<int> host_allocation(LinearAllocs::hipHostMalloc,
                                          device_allocation.width() * height);
    benchmark.Run(device_allocation.ptr(), device_allocation.pitch(), host_allocation.ptr(),
                  device_allocation.width(), device_allocation.width(), device_allocation.height(),
                  hipMemcpyHostToDevice, stream);
  } else if (kind == hipMemcpyHostToHost) {
    LinearAllocGuard<int> src_allocation(LinearAllocs::hipHostMalloc, width * sizeof(int) * height);
    LinearAllocGuard<int> dst_allocation(LinearAllocs::hipHostMalloc, width * sizeof(int) * height);
    benchmark.Run(dst_allocation.ptr(), width * sizeof(int), src_allocation.ptr(),
                  width * sizeof(int), width * sizeof(int), height, hipMemcpyHostToHost, stream);
  } else {
    // hipMemcpyDeviceToDevice
    int src_device = std::get<0>(GetDeviceIds(enable_peer_access));
    int dst_device = std::get<1>(GetDeviceIds(enable_peer_access));

    LinearAllocGuard2D<int> src_allocation(width, height);
    HIP_CHECK(hipSetDevice(dst_device));
    LinearAllocGuard2D<int> dst_allocation(width, height);

    HIP_CHECK(hipSetDevice(src_device));
    benchmark.Run(dst_allocation.ptr(), dst_allocation.pitch(), src_allocation.ptr(),
                  src_allocation.pitch(), dst_allocation.width(), dst_allocation.height(),
                  hipMemcpyDeviceToDevice, stream);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpy2DAsync` from Device to Host:
 *    -# Allocation size
 *      - Small: 4 KB x 32 B
 *      - Medium: 4 MB x 32 B
 *      - Large: 16 MB x 32 B
 * Test source
 * ------------------------
 * - performance/api/memcpy/hipMemcpy2DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipMemcpy2DAsync_DeviceToHost) {
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(width, 32, hipMemcpyDeviceToHost);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpy2DAsync` from Host to Device:
 *    -# Allocation size
 *      - Small: 4 KB x 32 B
 *      - Medium: 4 MB x 32 B
 *      - Large: 16 MB x 32 B
 * Test source
 * ------------------------
 * - performance/api/memcpy/hipMemcpy2DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipMemcpy2DAsync_HostToDevice) {
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(width, 32, hipMemcpyHostToDevice);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpy2DAsync` from Host to Host:
 *    -# Allocation size
 *      - Small: 4 KB x 32 B
 *      - Medium: 4 MB x 32 B
 *      - Large: 16 MB x 32 B
 * Test source
 * ------------------------
 * - performance/api/memcpy/hipMemcpy2DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipMemcpy2DAsync_HostToHost) {
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(width, 32, hipMemcpyHostToHost);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpy2DAsync` from Device to Device with peer access disabled:
 *    -# Allocation size
 *      - Small: 4 KB x 32 B
 *      - Medium: 4 MB x 32 B
 *      - Large: 16 MB x 32 B
 * Test source
 * ------------------------
 * - performance/api/memcpy/hipMemcpy2DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipMemcpy2DAsync_DeviceToDevice_DisablePeerAccess) {
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(width, 32, hipMemcpyDeviceToDevice);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpy2D` from Device to Device with peer access enabled:
 *    -# Allocation size
 *      - Small: 4 KB x 32 B
 *      - Medium: 4 MB x 32 B
 *      - Large: 16 MB x 32 B
 * Test source
 * ------------------------
 * - performance/api/memcpy/hipMemcpy2DAsync.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - Device supports Peer-to-Peer access
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipMemcpy2DAsync_DeviceToDevice_EnablePeerAccess) {
  if (HipTest::getDeviceCount() < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(width, 32, hipMemcpyDeviceToDevice, true);
}

// Copies rows padded to pitches 64 and 128 bytes wider than the row, starting src_offset and
// dst_offset bytes into their allocations
static void RunPaddedBenchmark(size_t width, size_t height, size_t src_offset, size_t dst_offset) {
  const size_t src_pitch = width + 64;
  const size_t dst_pitch = width + 128;
  const size_t src_size = src_offset + src_pitch * height;
  const size_t dst_size = dst_offset + dst_pitch * height;
  size_t free_memory = 0, total_memory = 0;
  HIP_CHECK(hipMemGetInfo(&free_memory, &total_memory));
  if (free_memory < src_size + dst_size + 1_MB) {
    HIP_SKIP_TEST("Not enough free device memory");
  }

  Memcpy2DAsyncBenchmark benchmark;
  benchmark.AddSectionName("(" + std::to_string(width) + ", " + std::to_string(height) + ")");
  benchmark.AddSectionName("offsets (" + std::to_string(src_offset) + ", " +
                           std::to_string(dst_offset) + ")");
  benchmark.RegisterBandwidth(width * height);

  const StreamGuard stream_guard(Streams::created);
  LinearAllocGuard<uint8_t> src_allocation(LinearAllocs::hipMalloc, src_size);
  LinearAllocGuard<uint8_t> dst_allocation(LinearAllocs::hipMalloc, dst_size);
  benchmark.Run(dst_allocation.ptr() + dst_offset, dst_pitch, src_allocation.ptr() + src_offset,
                src_pitch, width, height, hipMemcpyDeviceToDevice, stream_guard.stream());
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpy2DAsync` from Device to Device between rows padded to a pitch wider than
 *    the row, with source and destination rows that start 16-byte aligned, and 7 and 11 bytes past
 *    a 16-byte boundary:
 *    -# Copy size
 *      - Small: 64 B x 32
 *      - Medium: 4 KB x 128, 4 KB x 1024
 *      - Large, the size of the last-level cache of some devices: 16 KB x 16384
 * Test source
 * ------------------------
 * - performance/api/memcpy/hipMemcpy2DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Performance_hipMemcpy2DAsync_DeviceToDevice_Padded) {
  const auto [width, height] =
      GENERATE(table<size_t, size_t>({{64, 32}, {4_KB, 128}, {4_KB, 1024}, {16_KB, 16384}}));
  const auto [src_offset, dst_offset] = GENERATE(table<size_t, size_t>({{0, 0}, {7, 11}}));
  RunPaddedBenchmark(width, height, src_offset, dst_offset);
}

/**
 * End doxygen group memcpy.
 * @}
 */
