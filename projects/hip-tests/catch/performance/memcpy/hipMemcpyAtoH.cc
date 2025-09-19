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

class MemcpyAtoHBenchmark : public Benchmark<MemcpyAtoHBenchmark> {
 public:
  void operator()(void* dst, hipArray_t src_array, size_t allocation_size) {
    TIMED_SECTION(kTimerTypeCpu) { HIP_CHECK(hipMemcpyAtoH(dst, src_array, 0, allocation_size)); }
  }
};

static void RunBenchmark(LinearAllocs host_allocation_type, size_t width) {
  MemcpyAtoHBenchmark benchmark;
  benchmark.AddSectionName(std::to_string(width));
  benchmark.AddSectionName(GetAllocationSectionName(host_allocation_type));

  size_t allocation_size = width * sizeof(int);
  LinearAllocGuard<int> host_allocation(host_allocation_type, allocation_size);
  ArrayAllocGuard<int> array_allocation(make_hipExtent(width, 0, 0), hipArrayDefault);
  benchmark.Run(host_allocation.ptr(), array_allocation.ptr(), allocation_size);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyAtoH` from Device array to Host:
 *    -# Allocation size
 *      - Small: 512 B
 *      - Medium: 1024 B
 *      - Large: 4096 B
 *    -# Allocation type
 *      - Host: host pinned and pageable
 * Test source
 * ------------------------
 *  - performance/memcpy/hipMemcpyAtoH.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpyAtoH") {
  CHECK_IMAGE_SUPPORT

  const auto allocation_size = GENERATE(512, 1024, 4096);
  const auto host_allocation_type = GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc);
  RunBenchmark(host_allocation_type, allocation_size);
}

/**
 * End doxygen group memcpy.
 * @}
 */
