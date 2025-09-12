/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <chrono>
__global__ void mallocTest() {
  size_t size = 123;
  char* ptr = (char*)malloc(size);
  memset(ptr, 0, size);
  free(ptr);
}
__global__ void mallocTest_1() {
  size_t size = 1024;
  int* ptr = (int*)malloc(size);
  memset(ptr, 0, size);
  free(ptr);
}
/**
 * The tests in this file are added to see the performance improvement with the
 * Improve launch perf for Device Heap kernels task : SWDEV-513197
 */
/**
 * @addtogroup hipLaunchKernelGGL hipLaunchKernelGGL
 * @{
 * @ingroup PerformanceTest
 */
/**
 * Test Description
 * ------------------------
 * - This test case, tests the following scenario :
 * - 1) Create kernel call
 * - 2) In the kernel allocoate device heap memory
 * - 3) If any kernel uses device heap, the launch needs to be preceeded by an
        init kernel, Save on the extra barrier packet launch/flush between the
        init heap kernel and user kernel
 * - 4) Capture above Kernel Latency.
 * - 5) Kernel Laterncy has to be improved with the feature
 *      'Improve launch perf for Device Heap kernels. Task : SWDEV-513197'
 * - 6) Sencond Kernel Latency will be less compared to first
 *      launch latency due to the absense of  init kernel launch.
 * Test source
 * ------------------------
 * - catch/perftests/memory/hipPerfDeviceHeapMemory.cc
 *
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.5
 */
TEST_CASE("Unit_Perf_Device_Heap_Memory_Allocation") {
  HIP_CHECK(hipDeviceSetLimit(hipLimitMallocHeapSize, 128 * 1024 * 1024));
  hipEvent_t event;
  HIP_CHECK(hipEventCreate(&event));
  REQUIRE(event != nullptr);
  hipStream_t stream{nullptr};
  HIP_CHECK(hipStreamCreate(&stream));
  REQUIRE(stream != nullptr);
  HIP_CHECK(hipEventRecord(event, stream));
  HIP_CHECK(hipEventSynchronize(event));
  HIP_CHECK(hipStreamSynchronize(stream));
  // First Kernel Launch
  auto start = std::chrono::system_clock::now();
  mallocTest<<<1, 5, 0, stream>>>();
  auto end = std::chrono::system_clock::now();
  // Second Kernel Launch
  mallocTest_1<<<1, 5, 0, stream>>>();
  HIP_CHECK(hipDeviceSynchronize());
  auto end_1 = std::chrono::system_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  auto time_1 = std::chrono::duration_cast<std::chrono::microseconds>(end_1 - end).count();
  REQUIRE(time > time_1);
  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK(hipStreamDestroy(stream));
  std::cout << "First Kernel Latency: " << time << " micro seconds" << std::endl;
  std::cout << "Second Kernel Latency: " << time_1 << " micro seconds" << std::endl;
}
/**
 * End doxygen group PerformanceTest.
 * @}
 */
