/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
__global__ void Sum(void* ptr, size_t size) {
  size_t index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index != 0 && index < size) {
    atomicAdd(&((unsigned long long*)ptr)[0], ((unsigned long long*)ptr)[index]);
  }
}
class MemcpyHtoDKernelDtoHv1AsyncBenchmark
    : public Benchmark<MemcpyHtoDKernelDtoHv1AsyncBenchmark> {
 public:
  void operator()(void* host_mem, void* device_mem, size_t size, const hipStream_t& stream) {
    size_t count = size / sizeof(size_t);
    for (size_t i = 0; i < count; i++) {
      ((size_t*)host_mem)[i] = i;
    }
    TIMED_SECTION_STREAM(kTimerTypeCpu, stream) {
      HIP_CHECK(
          hipMemcpyHtoDAsync(reinterpret_cast<hipDeviceptr_t>(device_mem), host_mem, size, stream));
      int threads_num = 32;
      Sum<<<count / threads_num + 1, threads_num, 0, stream>>>(device_mem, count);
      HIP_CHECK(
          hipMemcpyDtoHAsync(host_mem, reinterpret_cast<hipDeviceptr_t>(device_mem), size, stream));
      HIP_CHECK(hipStreamSynchronize(stream));
    }
    size_t sum = ((size_t*)host_mem)[0];
    REQUIRE(sum == count * (count - 1) / 2);
  }
};
class MemcpyHtoDKernelDtoHv2AsyncBenchmark
    : public Benchmark<MemcpyHtoDKernelDtoHv2AsyncBenchmark> {
 public:
  void operator()(void* host_mem, void* device_mem, size_t size, const hipStream_t& stream) {
    size_t count = size / sizeof(size_t);
    for (size_t i = 0; i < count; i++) {
      ((size_t*)host_mem)[i] = i;
    }
    TIMED_SECTION_STREAM(kTimerTypeCpu, stream) {
      HIP_CHECK(hipMemcpyAsync(device_mem, host_mem, size, hipMemcpyHostToDevice, stream));
      int threads_num = 32;
      Sum<<<count / threads_num + 1, threads_num, 0, stream>>>(device_mem, count);
      HIP_CHECK(hipMemcpyWithStream(host_mem, device_mem, size, hipMemcpyDeviceToHost, stream));
      HIP_CHECK(hipStreamSynchronize(stream));
    }
    size_t sum = ((size_t*)host_mem)[0];
    REQUIRE(sum == count * (count - 1) / 2);
  }
};
template <typename BenchmarkType> static void RunBenchmark(LinearAllocs host_allocation_type,
                                                           LinearAllocs device_allocation_type,
                                                           size_t size) {
  BenchmarkType benchmark;
  if (size < 1_KB) {
    benchmark.AddSectionName(std::to_string(size));
  } else if (size < 1_MB) {
    benchmark.AddSectionName(std::to_string(size / 1024) + std::string(" KB"));
  } else {
    benchmark.AddSectionName(std::to_string(size / (1024 * 1024)) + std::string(" MB"));
  }
  benchmark.AddSectionName(GetAllocationSectionName(host_allocation_type));
  const StreamGuard stream_guard(Streams::created);
  const hipStream_t stream = stream_guard.stream();
  LinearAllocGuard<size_t> device_allocation(device_allocation_type, size);
  LinearAllocGuard<size_t> host_allocation(host_allocation_type, size);
  benchmark.Run(host_allocation.ptr(), device_allocation.ptr(), size, stream);
}
/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyHtoDAsync->Kernel->hipMemcpyDtoHAsync` from Device to Host:
 *    -# Allocation size
 *      - Small: 4 KB
 *      - Medium: 4 MB
 *      - Large: 16 MB
 *    -# Allocation type
 *      - Source: device malloc
 *      - Destination: host pinned and pageable
 * Test source
 * ------------------------
 *  - performance/memcpy/hipMemcpyDtoHAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpyHtoDKernelDtoHV1Async") {
  const auto allocation_size =
      GENERATE(16, 128, 1_KB, 4_KB, 16_KB, 256_KB, 512_KB, 1_MB, 4_MB, 16_MB, 128_MB);
  const auto device_allocation_type = LinearAllocs::hipMalloc;
  const auto host_allocation_type = GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc);
  RunBenchmark<MemcpyHtoDKernelDtoHv1AsyncBenchmark>(host_allocation_type, device_allocation_type,
                                                     allocation_size);
}
TEST_CASE("Performance_hipMemcpyHtoDKernelDtoHV2Async") {
  const auto allocation_size =
      GENERATE(16, 128, 1_KB, 4_KB, 16_KB, 256_KB, 512_KB, 1_MB, 4_MB, 16_MB, 128_MB);
  const auto device_allocation_type = LinearAllocs::hipMalloc;
  const auto host_allocation_type = GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc);
  RunBenchmark<MemcpyHtoDKernelDtoHv2AsyncBenchmark>(host_allocation_type, device_allocation_type,
                                                     allocation_size);
}
/**
 * End doxygen group memcpy.
 * @}
 */
