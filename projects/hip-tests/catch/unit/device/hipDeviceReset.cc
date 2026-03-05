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

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>

/**
 * @addtogroup hipDeviceReset hipDeviceReset
 * @{
 * @ingroup DeviceTest
 * `hipDeviceReset(void)` -
 * The state of current device is discarded and updated to a fresh state.
 *
 * Calling this function deletes all streams created, memory allocated, kernels running, events
 * created. Make sure that no other thread is using the device or streams, memory, kernels, events
 * associated with the current device.
 */

/**
 * Test Description
 * ------------------------
 *  - Validates that device reset frees allocated memory and
 *    reverts modified flags and configs to its default values.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceReset.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipDeviceReset_Positive_Basic") {
  const auto device = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(device));

  INFO("Current device is: " << device);
  HIP_CHECK(hipDeviceReset());

  unsigned int flags_before = 0u;
  HIP_CHECK(hipGetDeviceFlags(&flags_before));
  hipSharedMemConfig mem_config_before;
  HIP_CHECK(hipDeviceGetSharedMemConfig(&mem_config_before));

  void* ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, 500));

  const auto cache_config_ret = hipDeviceSetCacheConfig(hipFuncCachePreferL1);
  REQUIRE((cache_config_ret == hipSuccess || cache_config_ret == hipErrorNotSupported));

  const auto shared_mem_config_ret = hipDeviceSetSharedMemConfig(
      mem_config_before == hipSharedMemBankSizeFourByte ? hipSharedMemBankSizeEightByte
                                                        : hipSharedMemBankSizeFourByte);
  REQUIRE((shared_mem_config_ret == hipSuccess || shared_mem_config_ret == hipErrorNotSupported));

  HIP_CHECK(hipSetDeviceFlags(hipDeviceScheduleBlockingSync));

  HIP_CHECK(hipDeviceReset());

  unsigned int flags_after = 0u;
  CHECK(hipGetDeviceFlags(&flags_after) == hipSuccess);
  CHECK(flags_after == flags_before);

  // This will faill in ASAN due to how we handle free
#if !defined(ENABLE_ADDRESS_SANITIZER)
  CHECK(hipFree(ptr) == hipErrorInvalidValue);
#endif

  if (cache_config_ret == hipSuccess) {
    hipFuncCache_t cache_config;
    CHECK(hipDeviceGetCacheConfig(&cache_config) == hipSuccess);
    CHECK(cache_config == hipFuncCachePreferNone);
  }

  if (shared_mem_config_ret == hipSuccess) {
    hipSharedMemConfig mem_config_after;
    CHECK(hipDeviceGetSharedMemConfig(&mem_config_after) == hipSuccess);
    CHECK(mem_config_after == mem_config_before);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Resets device from another thread
 *  - Validates that device reset frees allocated memory from the main
 *    thread, and reverts modified flags and configs to its default values.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceReset.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipDeviceReset_Positive_Threaded") {
  HIP_CHECK(hipSetDevice(0));
  INFO("Current device is: " << 0);
  HIP_CHECK(hipDeviceReset());

  unsigned int flags_before = 0u;
  HIP_CHECK(hipGetDeviceFlags(&flags_before));
  hipSharedMemConfig mem_config_before;
  HIP_CHECK(hipDeviceGetSharedMemConfig(&mem_config_before));

  void* ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, 500));

  const auto cache_config_ret = hipDeviceSetCacheConfig(hipFuncCachePreferL1);
  REQUIRE((cache_config_ret == hipSuccess || cache_config_ret == hipErrorNotSupported));

  const auto shared_mem_config_ret = hipDeviceSetSharedMemConfig(
      mem_config_before == hipSharedMemBankSizeFourByte ? hipSharedMemBankSizeEightByte
                                                        : hipSharedMemBankSizeFourByte);
  REQUIRE((shared_mem_config_ret == hipSuccess || shared_mem_config_ret == hipErrorNotSupported));


  HIP_CHECK(hipSetDeviceFlags(hipDeviceScheduleBlockingSync));

  std::thread([] {
    HIP_CHECK_THREAD(hipSetDevice(0));
    HIP_CHECK_THREAD(hipDeviceReset());
  }).join();
  HIP_CHECK_THREAD_FINALIZE();

  unsigned int flags_after = 0u;
  CHECK(hipGetDeviceFlags(&flags_after) == hipSuccess);
  CHECK(flags_after == flags_before);

#if !defined(ENABLE_ADDRESS_SANITIZER)
  CHECK(hipFree(ptr) == hipErrorInvalidValue);
#endif

  if (cache_config_ret == hipSuccess) {
    hipFuncCache_t cache_config;
    CHECK(hipDeviceGetCacheConfig(&cache_config) == hipSuccess);
    CHECK(cache_config == hipFuncCachePreferNone);
  }

  if (shared_mem_config_ret == hipSuccess) {
    hipSharedMemConfig mem_config_after;
    CHECK(hipDeviceGetSharedMemConfig(&mem_config_after) == hipSuccess);
    CHECK(mem_config_after == mem_config_before);
  }
}

__global__ void allocOnly(void** out) {
  int* mem;
  if (threadIdx.x == 0) {
    mem = static_cast<int*>(malloc(sizeof(int)));
    *mem = 10; 
    // printf("Hello World %d\n", *mem);
    *out = mem;
  }

  // if (threadIdx.x == 0) {
  //   free(mem);
  // }
}

__global__ void freeOnly(void* mem) {

  if (threadIdx.x == 0) {
    free(mem);
  }
}

__global__ void allodAndFreef(int* outPtr, int* outPtr2) {
  if (threadIdx.x == 0) {
 	void* mem = malloc(sizeof(int));
    free(mem);
  }
}

// __global__ void allodAndFree(int* outPtr, int* outPtr2) {
//   int* mem;
//   if (threadIdx.x == 0) {
//     mem = static_cast<int*>(malloc(209));
//     *mem = 10; 
//     *outPtr = *mem;
//     // printf("Hello World %d\n", *mem);
//   }



//   if (threadIdx.x == 0) {
//   int* mem2 = static_cast<int*>(malloc(8192));
//   *mem2 = 20; 
//   *outPtr2 = *mem2;

//     if (threadIdx.x == 0) {
//     free(mem);
//   }

//     if (threadIdx.x == 0) {
//     free(mem2);
//   }
//   // printf("Hello World Again%d\n", *mem2);
// }
__global__ void allodAndFree() {
  int* mem;
  if (threadIdx.x == 0) {
    mem = static_cast<int*>(malloc(102400));
    *mem = 10; 
    free(mem);
  }
}

TEST_CASE("Unit_hipDeviceReset_LeakRegression") {
  void** memPtr;
  HIP_CHECK(hipMalloc(&memPtr, sizeof(void**)));

  HIP_CHECK(hipDeviceReset());

  hipLaunchKernelGGL((allodAndFree),  dim3(1) ,  dim3(1), 0 ,0);
  hipDeviceSynchronize();
  hipLaunchKernelGGL((allodAndFree),  dim3(1) ,  dim3(1), 0 ,0);
   hipDeviceSynchronize();
  HIP_CHECK(hipDeviceReset());

  hipLaunchKernelGGL((allodAndFree),  dim3(1) ,  dim3(1), 0 ,0);
  hipDeviceSynchronize();

  void* dev2;
  HIP_CHECK(hipMalloc(&dev2, sizeof(int)));


  // HIP_CHECK(hipFree(dev));
  HIP_CHECK(hipFree(dev2));
}

TEST_CASE("Unit_hipDeviceReset_LeakRegression_NoDoubleLaunch") {
  void** memPtr;
  HIP_CHECK(hipMalloc(&memPtr, sizeof(void**)));


  hipLaunchKernelGGL((allodAndFree),  dim3(1) ,  dim3(1), 0 ,0);
  hipDeviceSynchronize();
  hipLaunchKernelGGL((allodAndFree),  dim3(1) ,  dim3(1), 0 ,0);
   hipDeviceSynchronize();
  HIP_CHECK(hipDeviceReset());


  hipDeviceSynchronize();

  void* dev2;
  HIP_CHECK(hipMalloc(&dev2, sizeof(int)));


  // HIP_CHECK(hipFree(dev));
  HIP_CHECK(hipFree(dev2));
}

TEST_CASE("Unit_HostCall_DeviceAlloc_Basic") {
   int* memPtr;
  HIP_CHECK(hipMallocManaged(&memPtr, sizeof(int*)));
  int* memPtr2;
  HIP_CHECK(hipMallocManaged(&memPtr2, sizeof(int*)));

  

  hipLaunchKernelGGL((allodAndFree),  dim3(1) ,  dim3(1), 0 ,0);
  hipDeviceSynchronize();

  std::cout << "OUTPUT" << *memPtr << " " << *memPtr2 << std::endl;
  
  // hipLaunchKernelGGL((f),  dim3(1) ,  dim3(1), 0 , 0);
  // hipDeviceSynchronize();

  void* dev2;
  HIP_CHECK(hipMalloc(&dev2, sizeof(int)));


  HIP_CHECK(hipFree(memPtr));
   HIP_CHECK(hipFree(memPtr2));
  HIP_CHECK(hipFree(dev2));
}

/**
 * End doxygen group DeviceTest.
 * @}
 */
