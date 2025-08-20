/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "memcpy_performance_common.hh"

/**
 * @addtogroup memcpy memcpy
 * @{
 * @ingroup PerformanceTest
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
    if (src_device == -1 && dst_device == -1) {
      return;
    }

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
 *  - performance/memcpy/hipMemcpy3DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpy3DAsync_DeviceToHost") {
  CHECK_IMAGE_SUPPORT
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
 *  - performance/memcpy/hipMemcpy3DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpy3DAsync_HostToDevice") {
  CHECK_IMAGE_SUPPORT
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
 *  - performance/memcpy/hipMemcpy3DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpy3DAsync_HostToHost") {
  CHECK_IMAGE_SUPPORT
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
 *  - performance/memcpy/hipMemcpy3DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpy3DAsync_DeviceToDevice_DisablePeerAccess") {
  CHECK_IMAGE_SUPPORT
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
 *  - performance/memcpy/hipMemcpy3DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpy3DAsync_DeviceToDevice_EnablePeerAccess") {
  CHECK_IMAGE_SUPPORT
  if (HipTest::getDeviceCount() < 2) {
    HipTest::HIP_SKIP_TEST("This test requires 2 GPUs. Skipping.");
    return;
  }
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(make_hipExtent(width, 16, 4), hipMemcpyDeviceToDevice, true);
}

/**
 * End doxygen group memcpy.
 * @}
 */
