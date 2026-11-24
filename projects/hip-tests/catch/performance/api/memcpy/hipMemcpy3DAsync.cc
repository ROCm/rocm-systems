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

class Memcpy3DAsyncBenchmark : public Benchmark<Memcpy3DAsyncBenchmark> {
 public:
  void operator()(const hipPitchedPtr& dst_ptr, const hipPitchedPtr& src_ptr,
                  const hipExtent extent, hipMemcpyKind kind, const hipStream_t& stream) {
    hipMemcpy3DParms params = CreateMemcpy3DParam(dst_ptr, make_hipPos(0, 0, 0), src_ptr,
                                                  make_hipPos(0, 0, 0), extent, kind);
    TIMED_SECTION_STREAM(kTimerTypeEvent, stream) { HIP_CHECK(hipMemcpy3DAsync(&params, stream)); }
    HIP_CHECK(hipStreamSynchronize(stream));
  }
};

static void RunBenchmark(const hipExtent extent, hipMemcpyKind kind,
                         bool enable_peer_access = false) {
  Memcpy3DAsyncBenchmark benchmark;
  benchmark.AddSectionName("(" + std::to_string(extent.width) + ", " +
                           std::to_string(extent.height) + ", " + std::to_string(extent.depth) +
                           ")");

  const StreamGuard stream_guard(Streams::created);
  const hipStream_t stream = stream_guard.stream();

  if (kind == hipMemcpyDeviceToHost) {
    LinearAllocGuard3D<int> device_allocation(extent);
    LinearAllocGuard<int> host_allocation(
        LinearAllocs::hipHostMalloc,
        device_allocation.width() * device_allocation.height() * device_allocation.depth());
    benchmark.Run(make_hipPitchedPtr(host_allocation.ptr(), device_allocation.width(),
                                     device_allocation.width(), device_allocation.height()),
                  device_allocation.pitched_ptr(), device_allocation.extent(), kind, stream);
  } else if (kind == hipMemcpyHostToDevice) {
    LinearAllocGuard3D<int> device_allocation(extent);
    LinearAllocGuard<int> host_allocation(
        LinearAllocs::hipHostMalloc,
        device_allocation.pitch() * device_allocation.height() * device_allocation.depth());
    benchmark.Run(device_allocation.pitched_ptr(),
                  make_hipPitchedPtr(host_allocation.ptr(), device_allocation.pitch(),
                                     device_allocation.width(), device_allocation.height()),
                  device_allocation.extent(), kind, stream);
  } else if (kind == hipMemcpyHostToHost) {
    LinearAllocGuard3D<int> device_allocation(extent);
    LinearAllocGuard<int> src_allocation(LinearAllocs::hipHostMalloc,
                                         extent.width * extent.height * extent.depth);
    LinearAllocGuard<int> dst_allocation(LinearAllocs::hipHostMalloc,
                                         extent.width * extent.height * extent.depth);
    benchmark.Run(
        make_hipPitchedPtr(dst_allocation.ptr(), extent.width, extent.width, extent.height),
        make_hipPitchedPtr(src_allocation.ptr(), extent.width, extent.width, extent.height), extent,
        kind, stream);
  } else {
    // hipMemcpyDeviceToDevice
    int src_device = std::get<0>(GetDeviceIds(enable_peer_access));
    int dst_device = std::get<1>(GetDeviceIds(enable_peer_access));

    LinearAllocGuard3D<int> src_allocation(extent);
    HIP_CHECK(hipSetDevice(dst_device));
    LinearAllocGuard3D<int> dst_allocation(extent);
    HIP_CHECK(hipSetDevice(src_device));
    benchmark.Run(dst_allocation.pitched_ptr(), src_allocation.pitched_ptr(),
                  dst_allocation.extent(), kind, stream);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpy3DAsync` from Device to Host:
 *    -# Allocation size
 *      - Small: 4 KB x 16 B x 4 B
 *      - Medium: 4 MB x 16 B x 4 B
 *      - Large: 16 MB x 16 B x 4 B
 * Test source
 * ------------------------
 * - performance/api/memcpy/hipMemcpy3DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipMemcpy3DAsync_DeviceToHost) {
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(make_hipExtent(width, 16, 4), hipMemcpyDeviceToHost);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpy3DAsync` from Host to Device:
 *    -# Allocation size
 *      - Small: 4 KB x 16 B x 4 B
 *      - Medium: 4 MB x 16 B x 4 B
 *      - Large: 16 MB x 16 B x 4 B
 * Test source
 * ------------------------
 * - performance/api/memcpy/hipMemcpy3DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipMemcpy3DAsync_HostToDevice) {
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(make_hipExtent(width, 16, 4), hipMemcpyHostToDevice);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpy3DAsync` from Host to Host:
 *    -# Allocation size
 *      - Small: 4 KB x 16 B x 4 B
 *      - Medium: 4 MB x 16 B x 4 B
 *      - Large: 16 MB x 16 B x 4 B
 * Test source
 * ------------------------
 * - performance/api/memcpy/hipMemcpy3DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipMemcpy3DAsync_HostToHost) {
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(make_hipExtent(width, 16, 4), hipMemcpyHostToHost);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpy3DAsync` from Device to Device with peer access disabled:
 *    -# Allocation size
 *      - Small: 4 KB x 16 B x 4 B
 *      - Medium: 4 MB x 16 B x 4 B
 *      - Large: 16 MB x 16 B x 4 B
 * Test source
 * ------------------------
 * - performance/api/memcpy/hipMemcpy3DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipMemcpy3DAsync_DeviceToDevice_DisablePeerAccess) {
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(make_hipExtent(width, 16, 4), hipMemcpyDeviceToDevice);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpy3DAsync` from Device to Device with peer access enabled:
 *    -# Allocation size
 *      - Small: 4 KB x 16 B x 4 B
 *      - Medium: 4 MB x 16 B x 4 B
 *      - Large: 16 MB x 16 B x 4 B
 * Test source
 * ------------------------
 * - performance/api/memcpy/hipMemcpy3DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipMemcpy3DAsync_DeviceToDevice_EnablePeerAccess) {
  if (HipTest::getDeviceCount() < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(make_hipExtent(width, 16, 4), hipMemcpyDeviceToDevice, true);
}

// Copies rows padded to pitches 64 and 128 bytes wider than the row, in slices padded by a row,
// starting src_offset and dst_offset bytes into their allocations
static void RunPaddedBenchmark(const hipExtent extent, size_t src_offset, size_t dst_offset) {
  const size_t src_pitch = extent.width + 64;
  const size_t dst_pitch = extent.width + 128;
  const size_t rows_per_slice = extent.height + 1;
  const size_t src_size = src_offset + src_pitch * rows_per_slice * extent.depth;
  const size_t dst_size = dst_offset + dst_pitch * rows_per_slice * extent.depth;
  size_t free_memory = 0, total_memory = 0;
  HIP_CHECK(hipMemGetInfo(&free_memory, &total_memory));
  if (free_memory < src_size + dst_size + 1_MB) {
    HIP_SKIP_TEST("Not enough free device memory");
  }

  Memcpy3DAsyncBenchmark benchmark;
  benchmark.AddSectionName("(" + std::to_string(extent.width) + ", " +
                           std::to_string(extent.height) + ", " + std::to_string(extent.depth) +
                           ")");
  benchmark.AddSectionName("offsets (" + std::to_string(src_offset) + ", " +
                           std::to_string(dst_offset) + ")");
  benchmark.RegisterBandwidth(extent.width * extent.height * extent.depth);

  const StreamGuard stream_guard(Streams::created);
  LinearAllocGuard<uint8_t> src_allocation(LinearAllocs::hipMalloc, src_size);
  LinearAllocGuard<uint8_t> dst_allocation(LinearAllocs::hipMalloc, dst_size);
  benchmark.Run(make_hipPitchedPtr(dst_allocation.ptr() + dst_offset, dst_pitch, extent.width,
                                   rows_per_slice),
                make_hipPitchedPtr(src_allocation.ptr() + src_offset, src_pitch, extent.width,
                                   rows_per_slice),
                extent, hipMemcpyDeviceToDevice, stream_guard.stream());
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpy3DAsync` from Device to Device between rows padded to a pitch wider than
 *    the row and slices padded by a row, with source and destination rows that start 16-byte
 *    aligned, and 7 and 11 bytes past a 16-byte boundary:
 *    -# Copy size
 *      - Small: 257 B x 17 x 3
 *      - Medium: 1 KB x 1024 x 8
 *      - Large, the size of the last-level cache of some devices: 4 KB x 4096 x 16
 * Test source
 * ------------------------
 * - performance/api/memcpy/hipMemcpy3DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Performance_hipMemcpy3DAsync_DeviceToDevice_Padded) {
  const auto [width, height, depth] = GENERATE(
      table<size_t, size_t, size_t>({{257, 17, 3}, {1_KB, 1024, 8}, {4_KB, 4096, 16}}));
  const auto [src_offset, dst_offset] = GENERATE(table<size_t, size_t>({{0, 0}, {7, 11}}));
  RunPaddedBenchmark(make_hipExtent(width, height, depth), src_offset, dst_offset);
}

/**
 * End doxygen group memcpy.
 * @}
 */
