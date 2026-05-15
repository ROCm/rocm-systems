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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * @addtogroup hipHostRegister hipHostRegister
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipHostRegister (void *hostPtr, size_t sizeBytes, unsigned int flags)` -
 * register host memory so it can be accessed from the current device.
 */

#include <hip_test_common.hh>
#include <hip_test_helper.hh>
#include <hip_test_process.hh>
#include <hip_test_defgroups.hh>
#include "hip/hip_runtime_api.h"

#define INITIAL_VAL 1
#define EXPECTED_VAL 2
#define ADDITIONAL_MEMORY_PERCENT 10

static __global__ void Inc(uint8_t* Ad) {
  int tx = threadIdx.x + blockIdx.x * blockDim.x;
  Ad[tx] = Ad[tx] + 1;
}
/**
 * Test Description
 * ------------------------
 *    - Oversubscription: This testcase allocates host memory of size > total
 * GPU memory. Register the memory and try accessing it from kernel. Verify
 * the behaviour.
 * Test source
 * ------------------------
 *    - catch\unit\memory\hipHostRegister.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */
TEST_CASE("Stress_hipHostRegister_Oversubscription") {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::string arch = prop.gcnArchName;
#if HT_AMD
  if (std::string::npos == arch.find("xnack+")) {
    const char* msg = "Xnack not supported. Skipping test ..";
    HipTest::HIP_SKIP_TEST(msg);
    return;
  }
#endif
  size_t maxGpuMem = 0, availableMem = 0;
  // Get available GPU memory and total GPU memory
  HIP_CHECK(hipMemGetInfo(&availableMem, &maxGpuMem));
  INFO("Maximum GPU memory Size = " << maxGpuMem);
  size_t allocsize = maxGpuMem + ((maxGpuMem * ADDITIONAL_MEMORY_PERCENT) / 100);
  // Calculate grid size and block size
  size_t num_threads = prop.maxThreadsDim[0];
  size_t use_size = num_threads * 1024 * 1024;
  INFO("Chunk Size To Use = " << use_size);
  // Truncate the allocsize to multiples of use_size
  if (allocsize % use_size) {
    allocsize = allocsize - (allocsize % use_size);
  }
  INFO("Allocation Size = " << allocsize);
  // Get free host In bytes
  size_t hostMemFree = HipTest::getMemoryAmount() * 1024 * 1024;
  INFO("Free Host Memory = " << hostMemFree);
  // Ensure that allocsize < hostMemFree
  if (allocsize >= hostMemFree) {
    const char* msg = "Free Host Memory is insufficient. Skipping test ...";
    HipTest::HIP_SKIP_TEST(msg);
    return;
  }
  uint8_t* A = reinterpret_cast<uint8_t*>(malloc(allocsize));
  uint8_t* ptr;
  REQUIRE(A != NULL);
  // Inititalize
  memset(A, INITIAL_VAL, allocsize);
  // Register the entire host memory chunk
  HIP_CHECK(hipHostRegister(A, allocsize, 0));
  // Read and Write the entire allocsize in chunks of use_size
  for (size_t chk_chunk = 0; chk_chunk < allocsize; chk_chunk += use_size) {
    ptr = A + chk_chunk;
    hipLaunchKernelGGL(Inc, dim3(use_size / num_threads), num_threads, 0, 0, ptr);
    HIP_CHECK(hipDeviceSynchronize());
    for (int i = 0; i < use_size; i++) {
      REQUIRE(ptr[i] == EXPECTED_VAL);
    }
  }
  HIP_CHECK(hipHostUnregister(A));
  free(A);
}

/**
 * End doxygen group MemoryTest.
 * @}
 */
