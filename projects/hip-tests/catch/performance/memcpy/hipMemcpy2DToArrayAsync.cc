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

class Memcpy2DToArrayAsyncBenchmark : public Benchmark<Memcpy2DToArrayAsyncBenchmark> {
 public:
  void operator()(hipArray_t dst, const void* src, size_t src_pitch, size_t width, size_t height,
                  hipMemcpyKind kind, const hipStream_t& stream) {
    TIMED_SECTION_STREAM(kTimerTypeEvent, stream) {
      HIP_CHECK(hipMemcpy2DToArrayAsync(dst, 0, 0, src, src_pitch, width, height, kind, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
  }
};

static void RunBenchmark(size_t width, size_t height, hipMemcpyKind kind,
                         bool enable_peer_access = false) {
  Memcpy2DToArrayAsyncBenchmark benchmark;
  benchmark.AddSectionName("(" + std::to_string(width) + ", " + std::to_string(height) + ")");

  const StreamGuard stream_guard(Streams::created);
  const hipStream_t stream = stream_guard.stream();

  if (kind == hipMemcpyHostToDevice) {
    size_t allocation_size = width * height * sizeof(int);
    LinearAllocGuard<int> host_allocation(LinearAllocs::hipHostMalloc, allocation_size);
    ArrayAllocGuard<int> array_allocation(make_hipExtent(width, height, 0), hipArrayDefault);
    benchmark.Run(array_allocation.ptr(), host_allocation.ptr(), width * sizeof(int),
                  width * sizeof(int), height, hipMemcpyHostToDevice, stream);
  } else {
    // hipMemcpyDeviceToDevice
    int src_device = std::get<0>(GetDeviceIds(enable_peer_access));
    int dst_device = std::get<1>(GetDeviceIds(enable_peer_access));
    if (src_device == -1 && dst_device == -1) {
      return;
    }

    LinearAllocGuard2D<int> device_allocation(width, height);
    HIP_CHECK(hipSetDevice(dst_device));
    ArrayAllocGuard<int> array_allocation(make_hipExtent(width, height, 0), hipArrayDefault);
    HIP_CHECK(hipSetDevice(src_device));
    benchmark.Run(array_allocation.ptr(), device_allocation.ptr(), device_allocation.pitch(),
                  device_allocation.width(), device_allocation.height(), hipMemcpyDeviceToDevice,
                  stream);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpy2DToArrayAsync` from Host to Device:
 *    -# Allocation size
 *      - Small: 4 KB x 32 B
 *      - Medium: 8 KB x 32 B
 *      - Large: 16 KB x 32 B
 * Test source
 * ------------------------
 *  - performance/memcpy/hipMemcpy2DToArrayAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpy2DToArrayAsync_HostToDevice") {
  CHECK_IMAGE_SUPPORT

  const auto width = GENERATE(4_KB, 8_KB, 16_KB);
  RunBenchmark(width, 32, hipMemcpyHostToDevice);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpy2DToArrayAsync` from Device to Device with peer access disabled:
 *    -# Allocation size
 *      - Small: 4 KB x 32 B
 *      - Medium: 8 KB x 32 B
 *      - Large: 16 KB x 32 B
 * Test source
 * ------------------------
 *  - performance/memcpy/hipMemcpy2DToArrayAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpy2DToArrayAsync_DeviceToDevice_DisablePeerAccess") {
  CHECK_IMAGE_SUPPORT

  const auto width = GENERATE(4_KB, 8_KB, 16_KB);
  RunBenchmark(width, 32, hipMemcpyDeviceToDevice);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpy2DToArrayAsync` from Device to Device with peer access enabled:
 *    -# Allocation size
 *      - Small: 4 KB x 32 B
 *      - Medium: 8 KB x 32 B
 *      - Large: 16 KB x 32 B
 * Test source
 * ------------------------
 *  - performance/memcpy/hipMemcpy2DToArrayAsync.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - Device supports Peer-to-Peer access
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpy2DToArrayAsync_DeviceToDevice_EnablePeerAccess") {
  CHECK_IMAGE_SUPPORT

  if (HipTest::getDeviceCount() < 2) {
    HipTest::HIP_SKIP_TEST("This test requires 2 GPUs. Skipping.");
    return;
  }
  const auto width = GENERATE(4_KB, 8_KB, 16_KB);
  RunBenchmark(width, 32, hipMemcpyDeviceToDevice, true);
}

/**
 * End doxygen group memcpy.
 * @}
 */
