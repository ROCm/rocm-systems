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

class MemcpyHtoDBenchmark : public Benchmark<MemcpyHtoDBenchmark> {
 public:
  void operator()(hipDeviceptr_t& dst, void* src, size_t size) {
    TIMED_SECTION(kTimerTypeCpu) { HIP_CHECK(hipMemcpyHtoD(dst, src, size)); }
  }
};

static void RunBenchmark(LinearAllocs host_allocation_type, LinearAllocs device_allocation_type,
                         size_t size) {
  MemcpyHtoDBenchmark benchmark;
  benchmark.AddSectionName(std::to_string(size));
  benchmark.AddSectionName(GetAllocationSectionName(host_allocation_type));

  LinearAllocGuard<int> device_allocation(device_allocation_type, size);
  LinearAllocGuard<int> host_allocation(host_allocation_type, size);
  benchmark.Run(reinterpret_cast<hipDeviceptr_t>(device_allocation.ptr()), host_allocation.ptr(),
                size);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyHtoD` from Host to Device:
 *    -# Allocation size
 *      - Small: 4 KB
 *      - Medium: 4 MB
 *      - Large: 16 MB
 *    -# Allocation type
 *      - Source: host pinned and pageable
 *      - Destination: device malloc
 * Test source
 * ------------------------
 *  - performance/memcpy/hipMemcpyHtoD.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpyHtoD") {
  const auto allocation_size = GENERATE(4_KB, 4_MB, 16_MB);
  const auto device_allocation_type = LinearAllocs::hipMalloc;
  const auto host_allocation_type = GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc);
  RunBenchmark(host_allocation_type, device_allocation_type, allocation_size);
}

/**
 * End doxygen group memcpy.
 * @}
 */
