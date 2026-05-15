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

class MemcpyParam2DBenchmark : public Benchmark<MemcpyParam2DBenchmark> {
 public:
  void operator()(void* dst, size_t dst_pitch, void* src, size_t src_pitch, size_t width,
                  size_t height, hipMemcpyKind kind) {
    hip_Memcpy2D params = CreateMemcpy2DParam(dst, dst_pitch, src, src_pitch, width, height, kind);
    TIMED_SECTION(kTimerTypeCpu) { HIP_CHECK(hipMemcpyParam2D(&params)); }
  }
};

static void RunBenchmark(size_t width, size_t height, hipMemcpyKind kind,
                         bool enable_peer_access = false) {
  MemcpyParam2DBenchmark benchmark;
  benchmark.AddSectionName("(" + std::to_string(width) + ", " + std::to_string(height) + ")");

  if (kind == hipMemcpyDeviceToHost) {
    LinearAllocGuard2D<int> device_allocation(width, height);
    LinearAllocGuard<int> host_allocation(LinearAllocs::hipHostMalloc,
                                          device_allocation.width() * height);
    benchmark.Run(host_allocation.ptr(), device_allocation.width(), device_allocation.ptr(),
                  device_allocation.pitch(), device_allocation.width(), device_allocation.height(),
                  kind);
  } else if (kind == hipMemcpyHostToDevice) {
    LinearAllocGuard2D<int> device_allocation(width, height);
    LinearAllocGuard<int> host_allocation(LinearAllocs::hipHostMalloc,
                                          device_allocation.width() * height);
    benchmark.Run(device_allocation.ptr(), device_allocation.pitch(), host_allocation.ptr(),
                  device_allocation.width(), device_allocation.width(), device_allocation.height(),
                  kind);
  } else if (kind == hipMemcpyHostToHost) {
    LinearAllocGuard<int> src_allocation(LinearAllocs::hipHostMalloc, width * sizeof(int) * height);
    LinearAllocGuard<int> dst_allocation(LinearAllocs::hipHostMalloc, width * sizeof(int) * height);
    benchmark.Run(dst_allocation.ptr(), width * sizeof(int), src_allocation.ptr(),
                  width * sizeof(int), width * sizeof(int), height, kind);
  } else {
    // hipMemcpyDeviceToDevice
    int src_device = std::get<0>(GetDeviceIds(enable_peer_access));
    int dst_device = std::get<1>(GetDeviceIds(enable_peer_access));
    if (src_device == -1 && dst_device == -1) {
      return;
    }

    LinearAllocGuard2D<int> src_allocation(width, height);
    HIP_CHECK(hipSetDevice(dst_device));
    LinearAllocGuard2D<int> dst_allocation(width, height);
    HIP_CHECK(hipSetDevice(src_device));

    benchmark.Run(dst_allocation.ptr(), dst_allocation.pitch(), src_allocation.ptr(),
                  src_allocation.pitch(), dst_allocation.width(), dst_allocation.height(), kind);
  }
}

#if HT_NVIDIA
/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyParam2D` from Device to Host:
 *    -# Allocation size
 *      - Small: 4 KB x 32 B
 *      - Medium: 4 MB x 32 B
 *      - Large: 16 MB x 32 B
 * Test source
 * ------------------------
 *  - performance/memcpy/hipMemcpyParam2D.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpyParam2D_DeviceToHost") {
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(width, 32, hipMemcpyDeviceToHost);
}
#endif

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyParam2D` from Host to Device:
 *    -# Allocation size
 *      - Small: 4 KB x 32 B
 *      - Medium: 4 MB x 32 B
 *      - Large: 16 MB x 32 B
 * Test source
 * ------------------------
 *  - performance/memcpy/hipMemcpyParam2D.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpyParam2D_HostToDevice") {
  CHECK_IMAGE_SUPPORT
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(width, 32, hipMemcpyHostToDevice);
}

#if HT_NVIDIA
/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyParam2D` from Host to Host:
 *    -# Allocation size
 *      - Small: 4 KB x 32 B
 *      - Medium: 4 MB x 32 B
 *      - Large: 16 MB x 32 B
 * Test source
 * ------------------------
 *  - performance/memcpy/hipMemcpyParam2D.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpyParam2D_HostToHost") {
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(width, 32, hipMemcpyHostToHost);
}
#endif

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyParam2D` from Device to Device with peer access disabled:
 *    -# Allocation size
 *      - Small: 4 KB x 32 B
 *      - Medium: 4 MB x 32 B
 *      - Large: 16 MB x 32 B
 * Test source
 * ------------------------
 *  - performance/memcpy/hipMemcpyParam2D.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpyParam2D_DeviceToDevice_DisablePeerAccess") {
  CHECK_IMAGE_SUPPORT
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(width, 32, hipMemcpyDeviceToDevice);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyParam2D` from Device to Device with peer access enabled:
 *    -# Allocation size
 *      - Small: 4 KB x 32 B
 *      - Medium: 4 MB x 32 B
 *      - Large: 16 MB x 32 B
 * Test source
 * ------------------------
 *  - performance/memcpy/hipMemcpyParam2D.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - Device supports Peer-to-Peer access
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpyParam2D_DeviceToDevice_EnablePeerAccess") {
  CHECK_IMAGE_SUPPORT
  if (HipTest::getDeviceCount() < 2) {
    HipTest::HIP_SKIP_TEST("This test requires 2 GPUs. Skipping.");
    return;
  }
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(width, 32, hipMemcpyDeviceToDevice, true);
}

/**
 * End doxygen group memcpy.
 * @}
 */
