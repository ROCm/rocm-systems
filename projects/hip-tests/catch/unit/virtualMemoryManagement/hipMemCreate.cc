/*
Copyright (c) 2023-25 Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * @addtogroup hipMemCreate hipMemCreate
 * @{
 * @ingroup VirtualMemoryManagementTest
 * `hipError_t hipMemCreate (hipMemGenericAllocationHandle_t* handle,
 *                           size_t size,
 *                           const hipMemAllocationProp* prop,
 *                           unsigned long long flags)` -
 * Creates a memory allocation described by the properties and size.
 */

#include <hip_test_kernels.hh>
#include <hip_test_common.hh>

#include "hip_vmm_common.hh"

#define THREADS_PER_BLOCK 512
#define NUM_OF_BUFFERS 3
#define DATA_SIZE (1 << 13)

/**
 Kernel to perform Square of input data.
 */
static __global__ void square_kernel(int* Buff) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int temp = Buff[i] * Buff[i];
  Buff[i] = temp;
}

/**
 * Test Description
 * ------------------------
 *    - Allocate physical memories for different multiples of
 * granularity and deallocate them.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemCreate.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.0
 */
TEST_CASE("Unit_hipMemCreate_BasicAllocateDeAlloc_MultGranularity") {
  size_t granularity = 0;
  int deviceId = 0;
  CTX_CREATE();
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  hipMemAllocationProp prop{};

  SECTION("Memory Allocation Type as hipMemAllocationTypePinned") {
    prop.type = hipMemAllocationTypePinned;
  }

  #if HT_AMD
  SECTION("Memory Allocation Type as hipMemAllocationTypeUncached") {
    prop.type = hipMemAllocationTypeUncached;
  }
  #endif

  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;  // Current Devices
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  hipMemGenericAllocationHandle_t handle;
  // Allocate physical memory
  for (int mul = 1; mul < 64; mul++) {
    HIP_CHECK(hipMemCreate(&handle, granularity * mul, &prop, 0));
    HIP_CHECK(hipMemRelease(handle));
  }

  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Allocate physical memory and map it to virtual address range.
 * After setting device permission, copy data from host to VMM memory
 * and back to host. Verify the result. Release handle at end after
 * unmapping VMM range.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemCreate.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.0
 */
TEST_CASE("Unit_hipMemCreate_ChkDev2HstMemcpy_ReleaseHdlPostUnmap") {
  size_t granularity = 0;
  constexpr int N = DATA_SIZE;
  size_t buffer_size = N * sizeof(int);
  CTX_CREATE();
  int deviceId = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  hipMemAllocationProp prop{};

  SECTION("Memory Allocation Type as hipMemAllocationTypePinned") {
    prop.type = hipMemAllocationTypePinned;
  }

  #if HT_AMD
  SECTION("Memory Allocation Type as hipMemAllocationTypeUncached") {
    prop.type = hipMemAllocationTypeUncached;
  }
  #endif

  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;  // Current Devices
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
  hipMemGenericAllocationHandle_t handle;
  // Allocate physical memory
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  // Allocate virtual address range
  void* ptrA;
  HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));
  HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle, 0));
  // Set access
  hipMemAccessDesc accessDesc = {};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = device;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  // Make the address accessible to GPU 0
  HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));
  // Allocate host memory and intialize data
  std::vector<int> A_h(N), B_h(N);
  // Initialize with data
  for (size_t idx = 0; idx < N; idx++) {
    A_h[idx] = idx;
  }
  HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptrA), A_h.data(), buffer_size));
  HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptrA), buffer_size));
  REQUIRE(true == std::equal(B_h.begin(), B_h.end(), A_h.data()));
  HIP_CHECK(hipMemUnmap(ptrA, size_mem));
  HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
  HIP_CHECK(hipMemRelease(handle));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Allocate physical memory and map it to virtual address
 * range. After setting device permission, copy data from host
 * to VMM memory and back to host. Verify the result. Release
 * handle before the VMM range is used.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemCreate.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.0
 */
TEST_CASE("Unit_hipMemCreate_ChkDev2HstMemcpy_ReleaseHdlPreUse") {
  size_t granularity = 0;
  constexpr int N = DATA_SIZE;
  size_t buffer_size = N * sizeof(int);
  CTX_CREATE();
  int deviceId = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  hipMemAllocationProp prop{};

  SECTION("Memory Allocation Type as hipMemAllocationTypePinned") {
    prop.type = hipMemAllocationTypePinned;
  }

  #if HT_AMD
  SECTION("Memory Allocation Type as hipMemAllocationTypeUncached") {
    prop.type = hipMemAllocationTypeUncached;
  }
  #endif

  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;  // Current Devices
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
  hipMemGenericAllocationHandle_t handle;
  // Allocate physical memory
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  // Allocate virtual address range
  void* ptrA;
  HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));
  HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle, 0));
  HIP_CHECK(hipMemRelease(handle));
  // Set access
  hipMemAccessDesc accessDesc = {};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = device;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  // Make the address accessible to GPU 0
  HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));
  // Allocate host memory and intialize data
  std::vector<int> A_h(N), B_h(N);
  // Initialize with data
  for (size_t idx = 0; idx < N; idx++) {
    A_h[idx] = idx;
  }
  HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptrA), A_h.data(), buffer_size));
  HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptrA), buffer_size));
  REQUIRE(true == std::equal(B_h.begin(), B_h.end(), A_h.data()));
  HIP_CHECK(hipMemUnmap(ptrA, size_mem));
  HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Allocate physical memory and map it to virtual address
 * range. After setting device permission, copy data from host
 * to device, launch kernel to square the data, copy data back
 * to host. Verify the result.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemCreate.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.0
 */
TEST_CASE("Unit_hipMemCreate_ChkWithKerLaunch") {
  size_t granularity = 0;
  constexpr int N = DATA_SIZE;
  size_t buffer_size = N * sizeof(int);
  CTX_CREATE();
  int deviceId = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  hipMemAllocationProp prop{};

  SECTION("Memory Allocation Type as hipMemAllocationTypePinned") {
    prop.type = hipMemAllocationTypePinned;
  }

  #if HT_AMD
  SECTION("Memory Allocation Type as hipMemAllocationTypeUncached") {
    prop.type = hipMemAllocationTypeUncached;
  }
  #endif

  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;  // Current Devices
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
  hipMemGenericAllocationHandle_t handle;
  // Allocate physical memory
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  // Allocate virtual address range
  void* ptrA;
  HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));
  HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle, 0));
  HIP_CHECK(hipMemRelease(handle));
  // Set access
  hipMemAccessDesc accessDesc = {};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = device;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  // Make the address accessible to GPU 0
  HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));
  std::vector<int> A_h(N), B_h(N), C_h(N);
  // Initialize with data
  for (size_t idx = 0; idx < N; idx++) {
    A_h[idx] = idx;
    C_h[idx] = idx * idx;
  }
  HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptrA), A_h.data(), buffer_size));
  // Invoke kernel
  hipLaunchKernelGGL(square_kernel, dim3(N / THREADS_PER_BLOCK), dim3(THREADS_PER_BLOCK), 0, 0,
                     reinterpret_cast<int*>(ptrA));
  HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptrA), buffer_size));
  HIP_CHECK(hipDeviceSynchronize());
  REQUIRE(true == std::equal(B_h.begin(), B_h.end(), C_h.data()));
  HIP_CHECK(hipMemUnmap(ptrA, size_mem));
  HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Allocate multiple non-contiguous physical memory chunks
 * and map it to contiguous virtual address range. After setting
 * device permission, copy data from host to device, launch kernel
 * to square the data, copy data back to host. Verify the result.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemCreate.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.0
 */
TEST_CASE("Unit_hipMemCreate_MapNonContiguousChunks") {
  size_t granularity = 0;
  constexpr int numOfBuffers = NUM_OF_BUFFERS;
  constexpr int N = DATA_SIZE;
  size_t buffer_size = N * sizeof(int);
  CTX_CREATE();
  int deviceId = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  hipMemAllocationProp prop{};

  SECTION("Memory Allocation Type as hipMemAllocationTypePinned") {
    prop.type = hipMemAllocationTypePinned;
  }

  #if HT_AMD
  SECTION("Memory Allocation Type as hipMemAllocationTypeUncached") {
    prop.type = hipMemAllocationTypeUncached;
  }
  #endif

  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;  // Current Devices
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
  hipMemGenericAllocationHandle_t handle[NUM_OF_BUFFERS];
  // Allocate 3 physical memory chunks
  for (int count = 0; count < numOfBuffers; count++) {
    HIP_CHECK(hipMemCreate(&handle[count], size_mem, &prop, 0));
  }
  // Allocate virtual address range for all the memory chunks
  void* ptrA;
  HIP_CHECK(hipMemAddressReserve(&ptrA, (numOfBuffers * size_mem), 0, 0, 0));
  for (int idx = 0; idx < numOfBuffers; idx++) {
    unsigned long long uiptr = reinterpret_cast<unsigned long long>(ptrA);
    uiptr = uiptr + idx * size_mem;
    HIP_CHECK(hipMemMap(reinterpret_cast<void*>(uiptr), size_mem, 0, handle[idx], 0));
    HIP_CHECK(hipMemRelease(handle[idx]));
  }
  hipMemAccessDesc accessDesc = {};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = device;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  // Make the address accessible to GPU 0
  HIP_CHECK(hipMemSetAccess(ptrA, (numOfBuffers * size_mem), &accessDesc, 1));
  std::vector<int> A_h(numOfBuffers * size_mem), B_h(numOfBuffers * size_mem),
      C_h(numOfBuffers * size_mem);
  // Fill Data
  for (size_t idx = 0; idx < (numOfBuffers * N); idx++) {
    A_h[idx] = idx;
    C_h[idx] = idx * idx;
  }
  HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptrA), A_h.data(),
                          numOfBuffers * buffer_size));
  // Launch square kernel
  hipLaunchKernelGGL(square_kernel, dim3((N * numOfBuffers) / THREADS_PER_BLOCK),
                     dim3(THREADS_PER_BLOCK), 0, 0, reinterpret_cast<int*>(ptrA));
  HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptrA),
                         numOfBuffers * buffer_size));
  HIP_CHECK(hipDeviceSynchronize());
  // Validate Results
  REQUIRE(true == std::equal(B_h.begin(), B_h.end(), C_h.data()));
  for (int idx = 0; idx < numOfBuffers; idx++) {
    unsigned long long uiptr = reinterpret_cast<unsigned long long>(ptrA);
    uiptr = uiptr + idx * size_mem;
    HIP_CHECK(hipMemUnmap(reinterpret_cast<void*>(uiptr), size_mem));
  }
  HIP_CHECK(hipMemAddressFree(ptrA, (numOfBuffers * size_mem)));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - (Check if the VMM address can be memset) Map a physical chunk
 * to the VMM address range. Memset the VMM address range with initial
 * value. Validate.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemCreate.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.0
 */
TEST_CASE("Unit_hipMemCreate_ChkWithMemset") {
  size_t granularity = 0;
  constexpr int N = DATA_SIZE;
  size_t buffer_size = N * sizeof(int);
  constexpr int init_val = 0;
  CTX_CREATE();
  int deviceId = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  hipMemAllocationProp prop{};

  SECTION("Memory Allocation Type as hipMemAllocationTypePinned") {
    prop.type = hipMemAllocationTypePinned;
  }

  #if HT_AMD
  SECTION("Memory Allocation Type as hipMemAllocationTypeUncached") {
    prop.type = hipMemAllocationTypeUncached;
  }
  #endif

  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;  // Current Devices
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
  hipMemGenericAllocationHandle_t handle;
  // Allocate physical memory
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  // Allocate virtual address range
  void* ptrA;
  HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));
  HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle, 0));
  // Set access
  hipMemAccessDesc accessDesc = {};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = device;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  // Make the address accessible to GPU 0
  HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));
  std::vector<int> A_h(N);
  HIP_CHECK(hipMemset(ptrA, init_val, buffer_size));
  HIP_CHECK(hipMemcpyDtoH(A_h.data(), reinterpret_cast<hipDeviceptr_t>(ptrA), buffer_size));
  for (int idx = 0; idx < N; idx++) {
    REQUIRE(A_h[idx] == init_val);
  }
  HIP_CHECK(hipMemUnmap(ptrA, size_mem));
  HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
  HIP_CHECK(hipMemRelease(handle));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Negative Tests
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemCreate.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemCreate_Negative") {
  size_t granularity = 0;
  int deviceId = 0;
  hipDevice_t device;
  CTX_CREATE();
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  hipMemGenericAllocationHandle_t handle;
  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;  // Current Device
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));

  SECTION("Nullptr to handle") {
    REQUIRE(hipMemCreate(nullptr, granularity, &prop, 0) == hipErrorInvalidValue);
  }

  SECTION("Nullptr to prop") {
    REQUIRE(hipMemCreate(&handle, granularity, nullptr, 0) == hipErrorInvalidValue);
  }

  SECTION("pass size as 0") { REQUIRE(hipMemCreate(&handle, 0, &prop, 0) == hipErrorInvalidValue); }

  SECTION("Pass prop type as invalid") {
    prop.type = hipMemAllocationTypeInvalid;
    REQUIRE(hipMemCreate(&handle, granularity, &prop, 0) == hipErrorInvalidValue);
  }

  SECTION("pass location as invalid") {
    prop.location.type = hipMemLocationTypeInvalid;
    REQUIRE(hipMemCreate(&handle, granularity, &prop, 0) == hipErrorInvalidValue);
  }

  SECTION("non multiple of granularity") {
    REQUIRE(hipMemCreate(&handle, (granularity - 1), &prop, 0) == hipErrorInvalidValue);
  }

  SECTION("pass location id as -1") {
    prop.location.id = -1;  // set to non existing device
    REQUIRE(hipMemCreate(&handle, granularity, &prop, 0) == hipErrorInvalidDevice);
  }

  SECTION("pass location id as > highest device number") {
    int numDevices = 0;
    HIP_CHECK(hipGetDeviceCount(&numDevices));
    prop.location.id = numDevices;  // set to non existing device
    REQUIRE(hipMemCreate(&handle, granularity, &prop, 0) == hipErrorInvalidDevice);
  }

  CTX_DESTROY();
}

TEST_CASE("Unit_hipMemCreate_Capture") {
  CTX_CREATE();

  hipMemGenericAllocationHandle_t allocation_handle;
  size_t allocation_granularity = 0;
  constexpr int kDeviceId = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, kDeviceId));

  hipMemAllocationProp allocation_prop{};
  allocation_prop.type = hipMemAllocationTypePinned;
  allocation_prop.location.type = hipMemLocationTypeDevice;
  allocation_prop.location.id = device;

  HIP_CHECK(hipMemGetAllocationGranularity(&allocation_granularity, &allocation_prop,
                                           hipMemAllocationGranularityMinimum));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  GENERATE_CAPTURE();
  BEGIN_CAPTURE(stream);
  HIP_CHECK(hipMemCreate(&allocation_handle, allocation_granularity, &allocation_prop, 0));
  END_CAPTURE(stream);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipMemRelease(allocation_handle));
  CTX_DESTROY();
}


/**
 * End doxygen group VirtualMemoryManagementTest.
 * @}
 */
