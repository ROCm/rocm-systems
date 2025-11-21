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
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
   IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
 */

#include <hip_test_common.hh>
#include "mempool_common.hh"

/**
 * @addtogroup hipMemPoolCreate hipMemPoolCreate
 * @{
 * @ingroup StreamOTest
 * `hipMemPoolCreate(hipMemPool_t* mem_pool, const hipMemPoolProps* pool_props)` -
 * Creates a memory pool and returns the handle in mem pool
 */

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipMemPoolCreate behavior with invalid arguments:
 *    -# Nullptr mem_pool
 *    -# Nullptr props
 *    -# Invalid props alloc type
 *    -# Invalid props location type
 *    -# Invalid props location id
 *
 * Test source
 * ------------------------
 *  - /unit/memory/hipMemPoolCreate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemPoolCreate_Negative_Parameter") {
  checkMempoolSupported(0)

      int num_dev = 0;
  HIP_CHECK(hipGetDeviceCount(&num_dev));

  hipMemPoolProps pool_props;
  memset(&pool_props, 0, sizeof(pool_props));
  pool_props.allocType = hipMemAllocationTypePinned;
  pool_props.handleTypes = hipMemHandleTypeNone;
  pool_props.location.type = hipMemLocationTypeDevice;
  pool_props.location.id = 0;
  pool_props.win32SecurityAttributes = nullptr;

  hipMemPool_t mem_pool = nullptr;

  SECTION("Passing nullptr to mem_pool") {
    HIP_CHECK_ERROR(hipMemPoolCreate(nullptr, &pool_props), hipErrorInvalidValue);
  }

  SECTION("Passing nullptr to props") {
    HIP_CHECK_ERROR(hipMemPoolCreate(&mem_pool, nullptr), hipErrorInvalidValue);
  }

  SECTION("Passing invalid props alloc type") {
    pool_props.allocType = hipMemAllocationTypeInvalid;
    HIP_CHECK_ERROR(hipMemPoolCreate(&mem_pool, &pool_props), hipErrorInvalidValue);
    pool_props.allocType = hipMemAllocationTypePinned;
  }

  SECTION("Passing invalid props location type") {
    pool_props.location.type = hipMemLocationTypeInvalid;
    HIP_CHECK_ERROR(hipMemPoolCreate(&mem_pool, &pool_props), hipErrorInvalidValue);
    pool_props.location.type = hipMemLocationTypeDevice;
  }

  SECTION("Passing invalid props location id") {
    pool_props.location.id = num_dev;
    HIP_CHECK_ERROR(hipMemPoolCreate(&mem_pool, &pool_props), hipErrorInvalidValue);
    pool_props.location.id = 0;
  }
}

TEST_CASE("Unit_hipMemPoolCreate_With_maxSize") {
  checkMempoolSupported(0) hipMemPoolProps pool_props;
  memset(&pool_props, 0, sizeof(pool_props));
  pool_props.allocType = hipMemAllocationTypePinned;
  pool_props.handleTypes = hipMemHandleTypeNone;
  pool_props.location.type = hipMemLocationTypeDevice;
  pool_props.location.id = 0;
  pool_props.win32SecurityAttributes = nullptr;
#if HT_AMD
  pool_props.maxSize = 1024 * 1024 * 1024;
#endif
  float *A = nullptr, *B = nullptr;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  hipMemPool_t mem_pool = nullptr;
  HIP_CHECK(hipMemPoolCreate(&mem_pool, &pool_props));
  HIP_CHECK(
      hipMallocFromPoolAsync(reinterpret_cast<void**>(&A), 1024 * 1024 * 512, mem_pool, stream));
#if HT_AMD
  HIP_CHECK_ERROR(
      hipMallocFromPoolAsync(reinterpret_cast<void**>(&B), 1024 * 1024 * 513, mem_pool, stream),
      hipErrorOutOfMemory);
#else
  HIP_CHECK(
      hipMallocFromPoolAsync(reinterpret_cast<void**>(&B), 1024 * 1024 * 513, mem_pool, stream));
#endif
  HIP_CHECK(hipFreeAsync(A, stream));
  HIP_CHECK(hipMemPoolDestroy(mem_pool));
  HIP_CHECK(hipStreamDestroy(stream));
}

TEST_CASE("Unit_hipMemPoolCreate_Without_maxSize") {
  checkMempoolSupported(0) hipMemPoolProps pool_props;
  memset(&pool_props, 0, sizeof(pool_props));
  pool_props.allocType = hipMemAllocationTypePinned;
  pool_props.handleTypes = hipMemHandleTypeNone;
  pool_props.location.type = hipMemLocationTypeDevice;
  pool_props.location.id = 0;
  pool_props.win32SecurityAttributes = nullptr;

  float *A = nullptr, *B = nullptr;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  hipMemPool_t mem_pool = nullptr;
  HIP_CHECK(hipMemPoolCreate(&mem_pool, &pool_props));
  HIP_CHECK(
      hipMallocFromPoolAsync(reinterpret_cast<void**>(&A), 1024 * 1024 * 512, mem_pool, stream));
  HIP_CHECK(
      hipMallocFromPoolAsync(reinterpret_cast<void**>(&B), 1024 * 1024 * 513, mem_pool, stream));
  HIP_CHECK(hipMemPoolDestroy(mem_pool));
  HIP_CHECK(hipStreamDestroy(stream));
}

static __global__ void setKer(int* devptr) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  devptr[tid] = tid;
}
/**
 * Test Description
 * ------------------------
 *    - hipMemPoolCreate functionality tests
 * Create mempool for current device and other devices, if they exist, and
 * destroy them.
 * ------------------------
 *    - catch\unit\memory\hipMemPoolCreate.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemPoolCreate_DeviceTest") {
  checkMempoolSupported(0) int num_devices = 0;
  HIP_CHECK(hipGetDeviceCount(&num_devices));
  checkIfMultiDev(num_devices)
      // Scenario1
      SECTION("Simple Device Test") {
    for (int dev = 0; dev < num_devices; dev++) {
      hipMemPool_t mem_pool;
      hipMemPoolProps prop{};
      prop.allocType = hipMemAllocationTypePinned;
      prop.location.id = dev;
      prop.location.type = hipMemLocationTypeDevice;
      HIP_CHECK(hipMemPoolCreate(&mem_pool, &prop));
      HIP_CHECK(hipMemPoolDestroy(mem_pool));
    }
  }
  // Scenario2
  SECTION("Accessibility Test") {
    // Allocate a memory pool in current device
    constexpr int N = 1 << 12;
    constexpr int numThreadsPerBlk = 64;
    hipMemPool_t mem_pool;
    hipMemPoolProps prop{};
    prop.allocType = hipMemAllocationTypePinned;
    prop.location.id = 0;
    prop.location.type = hipMemLocationTypeDevice;
    HIP_CHECK(hipMemPoolCreate(&mem_pool, &prop));
    // Try allocating from mempool in other device context
    for (int dev = 1; dev < num_devices; dev++) {
      int* A_d;
      HIP_CHECK(hipSetDevice(dev));
      HIP_CHECK(
          hipMallocFromPoolAsync(reinterpret_cast<void**>(&A_d), N * sizeof(int), mem_pool, 0));
      HIP_CHECK(hipStreamSynchronize(0));
      HIP_CHECK(hipSetDevice(0));
      // Launch kernel to access A_d and free it on dev 0 context
      setKer<<<N / numThreadsPerBlk, numThreadsPerBlk, 0, 0>>>(A_d);
      HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(A_d), 0));
      HIP_CHECK(hipStreamSynchronize(0));
    }
    HIP_CHECK(hipMemPoolDestroy(mem_pool));
  }
}

/**
 * End doxygen group StreamOTest.
 * @}
 */
