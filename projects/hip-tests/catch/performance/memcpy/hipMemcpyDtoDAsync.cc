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

class MemcpyDtoDAsyncBenchmark : public Benchmark<MemcpyDtoDAsyncBenchmark> {
 public:
  void operator()(hipDeviceptr_t& dst, const hipDeviceptr_t& src, size_t size,
                  const hipStream_t& stream) {
    TIMED_SECTION_STREAM(kTimerTypeEvent, stream) {
      HIP_CHECK(hipMemcpyDtoDAsync(dst, src, size, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
  }
};

static void RunBenchmark(size_t size, bool enable_peer_access = false) {
  MemcpyDtoDAsyncBenchmark benchmark;
  benchmark.AddSectionName(std::to_string(size));

  const StreamGuard stream_guard(Streams::created);
  const hipStream_t stream = stream_guard.stream();
  int src_device = std::get<0>(GetDeviceIds(enable_peer_access));
  int dst_device = std::get<1>(GetDeviceIds(enable_peer_access));
  if (src_device == -1 && dst_device == -1) {
    return;
  }

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
 *  - Executes `hipMemcpyDtoDAsync` from Device to Device with peer access enabled:
 *    -# Allocation size
 *      - Small: 4 KB
 *      - Medium: 4 MB
 *      - Large: 16 MB
 *    -# Allocation type
 *      - Source: device malloc
 *      - Destination: device malloc
 * Test source
 * ------------------------
 *  - performance/memcpy/hipMemcpyDtoDAsync.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - Device supports Peer-to-Peer access
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpyDtoDAsync_PeerAccessEnabled") {
  if (HipTest::getDeviceCount() < 2) {
    HipTest::HIP_SKIP_TEST("This test requires 2 GPUs. Skipping.");
    return;
  }
  const auto allocation_size = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(allocation_size, true);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyDtoD` from Device to Device with peer access disabled:
 *    -# Allocation size
 *      - Small: 4 KB
 *      - Medium: 4 MB
 *      - Large: 16 MB
 *    -# Allocation type
 *      - Source: device malloc
 *      - Destination: device malloc
 * Test source
 * ------------------------
 *  - performance/memcpy/hipMemcpyDtoDAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpyDtoDAsync_PeerAccessDisabled") {
  const auto allocation_size = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(allocation_size);
}

/**
 * End doxygen group memcpy.
 * @}
 */
