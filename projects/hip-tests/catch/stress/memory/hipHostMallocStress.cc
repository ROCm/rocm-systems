/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <hip_test_helper.hh>

#define ADDITIONAL_MEMORY_PERCENT 10

// Stress allocation tests
// Try to allocate as much memory as possible
// But since max allocation can fail, we need to try the next value

TEST_CASE("Stress_hipHostMalloc_MaxAllocation") {
  size_t devMemAvail{0}, devMemFree{0};
  HIP_CHECK(hipMemGetInfo(&devMemFree, &devMemAvail));
  auto hostMemFree = HipTest::getMemoryAmount() * 1024 * 1024;  // In bytes
  REQUIRE(devMemFree > 0);
  REQUIRE(devMemAvail > 0);
  REQUIRE(hostMemFree > 0);
  // which is the limiter cpu or gpu
  size_t memFree = std::min(devMemFree, hostMemFree);
  char* d_ptr{nullptr};
  size_t counter{0};

  INFO("Max Allocation of " << memFree << " bytes!");
  while (hipHostMalloc(&d_ptr, memFree) != hipSuccess && memFree > 1) {
    counter++;
    INFO("Attempt to allocate " << memFree << " bytes out of " << devMemFree << "bytes Failed!");
    memFree >>= 1;          // reduce the memory to be allocated by half
    REQUIRE(counter <= 2);  // Make sure that we are atleast able to allocate
    // 1/4th of max memory
  }

  HIP_CHECK(hipMemset(d_ptr, 1, memFree));
  HIP_CHECK(hipDeviceSynchronize());  // Flush caches
  REQUIRE(std::all_of(d_ptr, d_ptr + memFree, [](unsigned char n) { return n == 1; }));
  HIP_CHECK(hipHostFree(d_ptr));
}

// Allocate more memory than total GPU memory in each available GPU.
// hipHostMalloc should return hipSuccess.

TEST_CASE("Stress_hipHostMalloc_MaxAllocation_AllGpu") {
  char* A = nullptr;
  size_t maxGpuMem = 0, availableMem = 0;
  int count = 0;
  HIP_CHECK(hipGetDeviceCount(&count));
  for (int dev = 0; dev < count; dev++) {
    // Get available GPU memory and total GPU memory
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipMemGetInfo(&availableMem, &maxGpuMem));
    size_t allocsize = maxGpuMem + ((maxGpuMem * ADDITIONAL_MEMORY_PERCENT) / 100);
    // Get free host In bytes
    size_t hostMemFree = HipTest::getMemoryAmount() * 1024 * 1024;
    if (allocsize < hostMemFree) {
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A), allocsize));
      // Check accessibility of memory
      constexpr size_t samplesize = 1024;
      constexpr int val = 32;
      // Write at beginning of memory chunk for a size of samplesize
      HIP_CHECK(hipMemset(A, val, samplesize));
      // Write at end of memory chunk for a size of samplesize
      HIP_CHECK(hipMemset((A + allocsize - 1 - samplesize), val, samplesize));
      HIP_CHECK(hipHostFree(A));
    } else {
      WARN("Skipping test as CPU memory is less than GPU memory");
    }
  }
}
