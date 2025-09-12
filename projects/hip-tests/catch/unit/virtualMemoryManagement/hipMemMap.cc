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

/**
 * @addtogroup hipMemMap hipMemMap
 * @{
 * @ingroup VirtualMemoryManagementTest
 * `hipError_t hipMemMap (void* ptr,
 *                        size_t size,
 *                        size_t offset,
 *                        hipMemGenericAllocationHandle_t handle,
 *                        unsigned long long flags)` -
 * Maps an allocation handle to a reserved virtual address range.
 */

#include <hip_test_common.hh>

#include "hip_vmm_common.hh"

constexpr int N = (1 << 13);
constexpr int num_buf = 3;
constexpr int initializer = 0;

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
 *    - Check if a physical chunk can be mapped/unmapped to same
 * vmm address range repeatedly. This test validates physical memory
 * euse using same vmm range.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemMap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemMap_SameMemoryReuse") {
  constexpr int iterations = 20;
  size_t granularity = 0;
  size_t buffer_size = N * sizeof(int);
  CTX_CREATE();
  int deviceId = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;  // Current Devices
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
  hipMemGenericAllocationHandle_t handle;
  // Allocate host memory and intialize data
  std::vector<int> A_h(N), B_h(N), C_h(N);
  // Initialize with data
  for (size_t idx = 0; idx < N; idx++) {
    A_h[idx] = idx;
    C_h[idx] = idx * idx;
  }
  // Allocate a physical memory chunk
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  // Allocate num_buf virtual address ranges
  hipDeviceptr_t ptrA;
  HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));
  hipMemAccessDesc accessDesc = {};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = device;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  for (int i = 0; i < iterations; i++) {
    std::fill(B_h.begin(), B_h.end(), initializer);
    HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle, 0));
    // Set access to GPU 0
    HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));
    HIP_CHECK(hipMemcpyHtoD(ptrA, A_h.data(), buffer_size));
    HIP_CHECK(hipMemcpyDtoH(B_h.data(), ptrA, buffer_size));
    REQUIRE(true == std::equal(B_h.begin(), B_h.end(), A_h.data()));
    square_kernel<<<dim3(N / threadsPerBlk), dim3(threadsPerBlk), 0, 0>>>(
        reinterpret_cast<int*>(ptrA));
    HIP_CHECK(hipMemcpyDtoH(B_h.data(), ptrA, buffer_size));
    HIP_CHECK(hipStreamSynchronize(0));
    REQUIRE(true == std::equal(B_h.begin(), B_h.end(), C_h.data()));
    HIP_CHECK(hipMemUnmap(ptrA, size_mem));
  }
  // Release resources
  HIP_CHECK(hipMemRelease(handle));
  HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Check if a physical chunk can be mapped/unmapped for multiple
 * vmm addresses. This test validates physical memory reuse using
 * different vmm ranges.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemMap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemMap_PhysicalMemoryReuse_SingleGPU") {
  size_t granularity = 0;
  size_t buffer_size = N * sizeof(int);
  CTX_CREATE();
  int deviceId = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;  // Current Devices
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
  hipMemGenericAllocationHandle_t handle;
  // Allocate host memory and intialize data
  std::vector<int> A_h(N), B_h(N), C_h(N);
  // Initialize with data
  for (size_t idx = 0; idx < N; idx++) {
    A_h[idx] = idx;
    C_h[idx] = idx * idx;
  }
  // Allocate a physical memory chunk
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  // Allocate num_buf virtual address ranges
  hipDeviceptr_t ptrA[num_buf];
  for (int buf = 0; buf < num_buf; buf++) {
    HIP_CHECK(hipMemAddressReserve(&ptrA[buf], size_mem, 0, 0, 0));
  }
  hipMemAccessDesc accessDesc = {};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = device;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  for (int buf = 0; buf < num_buf; buf++) {
    std::fill(B_h.begin(), B_h.end(), initializer);
    HIP_CHECK(hipMemMap(ptrA[buf], size_mem, 0, handle, 0));
    // Set access to GPU 0
    HIP_CHECK(hipMemSetAccess(ptrA[buf], size_mem, &accessDesc, 1));
    HIP_CHECK(hipMemcpyHtoD(ptrA[buf], A_h.data(), buffer_size));
    HIP_CHECK(hipMemcpyDtoH(B_h.data(), ptrA[buf], buffer_size));
    REQUIRE(true == std::equal(B_h.begin(), B_h.end(), A_h.data()));
    square_kernel<<<dim3(N / threadsPerBlk), dim3(threadsPerBlk), 0, 0>>>(
        reinterpret_cast<int*>(ptrA[buf]));
    HIP_CHECK(hipMemcpyDtoH(B_h.data(), ptrA[buf], buffer_size));
    HIP_CHECK(hipStreamSynchronize(0));
    REQUIRE(true == std::equal(B_h.begin(), B_h.end(), C_h.data()));
    HIP_CHECK(hipMemUnmap(ptrA[buf], size_mem));
  }
  // Release resources
  HIP_CHECK(hipMemRelease(handle));
  for (int buf = 0; buf < num_buf; buf++) {
    HIP_CHECK(hipMemAddressFree(ptrA[buf], size_mem));
  }

  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Check if a physical chunk can be mapped to multiple
 * vmm addresses at the same time and check data values integrity
 * between different VMMs.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemMap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemMap_PhysicalMemory_Map2MultVMMs") {
  size_t granularity = 0;
  size_t buffer_size = N * sizeof(int);
  CTX_CREATE();
  int deviceId = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;  // Current Devices
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
  hipMemGenericAllocationHandle_t handle;
  // Allocate host memory and intialize data
  std::vector<int> A_h(N), B_h(N);
  // Initialize with data
  for (size_t idx = 0; idx < N; idx++) {
    A_h[idx] = idx;
  }
  // Allocate a physical memory chunk
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  // Allocate num_buf virtual address ranges
  hipDeviceptr_t ptrA[num_buf];
  for (int buf = 0; buf < num_buf; buf++) {
    HIP_CHECK(hipMemAddressReserve(&ptrA[buf], size_mem, 0, 0, 0));
  }
  hipMemAccessDesc accessDesc = {};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = device;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  for (int buf = 0; buf < num_buf; buf++) {
    HIP_CHECK(hipMemMap(ptrA[buf], size_mem, 0, handle, 0));
  }
  // Set access for all the buffers.
  for (int buf = 0; buf < num_buf; buf++) {
    HIP_CHECK(hipMemSetAccess(ptrA[buf], size_mem, &accessDesc, 1));
  }
  // Copy data to VMM via ptrA[0]
  HIP_CHECK(hipMemcpyHtoD(ptrA[0], A_h.data(), buffer_size));
  // Validate the data contained in VMM using ptrA[0], ptrA[1],
  // ......, ptrA[num_buf-1]
  for (int buf = 0; buf < num_buf; buf++) {
    std::fill(B_h.begin(), B_h.end(), initializer);
    HIP_CHECK(hipMemcpyDtoH(B_h.data(), ptrA[buf], buffer_size));
    REQUIRE(true == std::equal(B_h.begin(), B_h.end(), A_h.data()));
  }

  for (int buf = 0; buf < num_buf; buf++) {
    HIP_CHECK(hipMemUnmap(ptrA[buf], size_mem));
  }

  // Release resources
  HIP_CHECK(hipMemRelease(handle));
  for (int buf = 0; buf < num_buf; buf++) {
    HIP_CHECK(hipMemAddressFree(ptrA[buf], size_mem));
  }

  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Check if a physical chunk can be mapped/unmapped for
 * multiple vmm addresses. This test validates physical memory
 * reuse using different vmm ranges on multiple devices.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemMap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemMap_PhysicalMemoryReuse_MultiDev") {
  CHECK_P2P_SUPPORT
  int devicecount = 0;
  HIP_CHECK(hipGetDeviceCount(&devicecount));
  if (devicecount < 2) {
    HipTest::HIP_SKIP_TEST("Machine is Single GPU. Skipping Test..");
    return;
  }
  size_t granularity = 0;
  size_t buffer_size = N * sizeof(int);
  for (int devX = 0; devX < devicecount; devX++) {
    hipDevice_t device;
    HIP_CHECK(hipDeviceGet(&device, devX));
    checkVMMSupported(device);
    hipMemAllocationProp prop{};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = device;  // Current Devices
    HIP_CHECK(
        hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
    REQUIRE(granularity > 0);
    size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
    hipMemGenericAllocationHandle_t handle;
    // Allocate host memory and intialize data
    std::vector<int> A_h(N), B_h(N);
    // Initialize with data
    for (size_t idx = 0; idx < N; idx++) {
      A_h[idx] = idx;
    }
    // Allocate a physical memory chunk
    HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
    // Allocate devicecount virtual address ranges
    std::vector<hipDeviceptr_t> ptrA(devicecount);
    for (int devY = 0; devY < devicecount; devY++) {
      HIP_CHECK(hipMemAddressReserve(&ptrA[devY], size_mem, 0, 0, 0));
    }
    for (int devY = 0; devY < devicecount; devY++) {
      hipDevice_t deviceToTest;
      HIP_CHECK(hipDeviceGet(&deviceToTest, devY));
      hipMemAccessDesc accessDesc = {};
      accessDesc.location.type = hipMemLocationTypeDevice;
      accessDesc.location.id = deviceToTest;
      accessDesc.flags = hipMemAccessFlagsProtReadWrite;
      HIP_CHECK(hipSetDevice(devY));
      std::fill(B_h.begin(), B_h.end(), initializer);
      HIP_CHECK(hipMemMap(ptrA[devY], size_mem, 0, handle, 0));
      // Set access to GPU 0
      HIP_CHECK(hipMemSetAccess(ptrA[devY], size_mem, &accessDesc, 1));
      HIP_CHECK(hipMemcpyHtoD(ptrA[devY], A_h.data(), buffer_size));
      HIP_CHECK(hipMemcpyDtoH(B_h.data(), ptrA[devY], buffer_size));
      REQUIRE(true == std::equal(B_h.begin(), B_h.end(), A_h.data()));
      HIP_CHECK(hipMemUnmap(ptrA[devY], size_mem));
    }
    HIP_CHECK(hipSetDevice(0));  // set the device back to 0.
    // Release resources
    HIP_CHECK(hipMemRelease(handle));
    for (int devY = 0; devY < devicecount; devY++) {
      HIP_CHECK(hipMemAddressFree(ptrA[devY], size_mem));
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - Check if different physical chunk can be mapped/unmapped
 * for single vmm address. This test validates VMM memory reuse
 * using different physical ranges.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemMap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemMap_VMMMemoryReuse_SingleGPU") {
  size_t granularity = 0;
  size_t buffer_size = N * sizeof(int);
  CTX_CREATE();
  int deviceId = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;  // Current Devices
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
  hipMemGenericAllocationHandle_t handle[num_buf];
  // Allocate host memory and intialize data
  std::vector<int> A_h(N), B_h(N), C_h(N);
  // Initialize with data
  for (size_t idx = 0; idx < N; idx++) {
    A_h[idx] = idx;
    C_h[idx] = idx * idx;
  }
  // Allocate a physical memory chunk
  for (int buf = 0; buf < num_buf; buf++) {
    HIP_CHECK(hipMemCreate(&handle[buf], size_mem, &prop, 0));
  }
  // Allocate num_buf virtual address ranges
  hipDeviceptr_t ptrA;
  HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));
  hipMemAccessDesc accessDesc = {};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = device;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  // Map ptrA to physical chunk
  for (int buf = 0; buf < num_buf; buf++) {
    std::fill(B_h.begin(), B_h.end(), initializer);
    HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle[buf], 0));
    // Set access to GPU 0
    HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));
    HIP_CHECK(hipMemcpyHtoD(ptrA, A_h.data(), buffer_size));
    HIP_CHECK(hipMemcpyDtoH(B_h.data(), ptrA, buffer_size));
    REQUIRE(true == std::equal(B_h.begin(), B_h.end(), A_h.data()));
#if HT_NVIDIA
    square_kernel<<<dim3(N / threadsPerBlk), dim3(threadsPerBlk), 0, 0>>>(
        reinterpret_cast<int*>(ptrA));
    HIP_CHECK(hipMemcpyDtoH(B_h.data(), ptrA, buffer_size));
    HIP_CHECK(hipStreamSynchronize(0));
    REQUIRE(true == std::equal(B_h.begin(), B_h.end(), C_h.data()));
#endif
    HIP_CHECK(hipMemUnmap(ptrA, size_mem));
  }
  // Release resources
  for (int buf = 0; buf < num_buf; buf++) {
    HIP_CHECK(hipMemRelease(handle[buf]));
  }
  HIP_CHECK(hipMemAddressFree(ptrA, size_mem));

  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Check if different physical chunk allocated in different devices
 * can be mapped/unmapped to single vmm address. This test validates VMM
 * memory reuse using different physical ranges.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemMap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemMap_VMMMemoryReuse_MultiGPU") {
  CHECK_P2P_SUPPORT
  int deviceId = 0, devicecount = 0;
  HIP_CHECK(hipGetDeviceCount(&devicecount));
  if (devicecount < 2) {
    HipTest::HIP_SKIP_TEST("Machine is Single GPU. Skipping Test..");
    return;
  }
  size_t granularity = 0;
  size_t buffer_size = N * sizeof(int);
  hipDevice_t device;
  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;  // Current Devices
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
  std::vector<hipMemGenericAllocationHandle_t> handle(devicecount);
  // Allocate host memory and intialize data
  std::vector<int> A_h(N), B_h(N);
  // Initialize with data
  for (size_t idx = 0; idx < N; idx++) {
    A_h[idx] = idx;
  }
  // Allocate a physical memory chunk
  for (int dev = 0; dev < devicecount; dev++) {
    hipDevice_t dev_handle;
    HIP_CHECK(hipDeviceGet(&dev_handle, dev));
    prop.location.id = dev_handle;
    HIP_CHECK(hipMemCreate(&handle[dev], size_mem, &prop, 0));
  }
  // Allocate devicecount virtual address ranges
  hipDeviceptr_t ptrA;
  HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));
  // Map ptrA to physical chunk
  SECTION("Set Access of VMM to Different GPU") {
    for (int dev = 0; dev < devicecount; dev++) {
      hipDevice_t device;
      HIP_CHECK(hipDeviceGet(&device, dev));
      hipMemAccessDesc accessDesc = {};
      accessDesc.location.type = hipMemLocationTypeDevice;
      accessDesc.location.id = device;
      accessDesc.flags = hipMemAccessFlagsProtReadWrite;
      HIP_CHECK(hipSetDevice(dev));
      std::fill(B_h.begin(), B_h.end(), initializer);
      HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle[dev], 0));
      HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));
      HIP_CHECK(hipMemcpyHtoD(ptrA, A_h.data(), buffer_size));
      HIP_CHECK(hipMemcpyDtoH(B_h.data(), ptrA, buffer_size));
      HIP_CHECK(hipMemUnmap(ptrA, size_mem));
      REQUIRE(true == std::equal(B_h.begin(), B_h.end(), A_h.data()));
    }
  }
  SECTION("Set Access of VMM to default GPU") {
    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type = hipMemLocationTypeDevice;
    accessDesc.location.id = device;
    accessDesc.flags = hipMemAccessFlagsProtReadWrite;
    for (int dev = 0; dev < devicecount; dev++) {
      std::fill(B_h.begin(), B_h.end(), initializer);
      HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle[dev], 0));
      HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));
      HIP_CHECK(hipMemcpyHtoD(ptrA, A_h.data(), buffer_size));
      HIP_CHECK(hipMemcpyDtoH(B_h.data(), ptrA, buffer_size));
      HIP_CHECK(hipMemUnmap(ptrA, size_mem));
      REQUIRE(true == std::equal(B_h.begin(), B_h.end(), A_h.data()));
    }
  }
  HIP_CHECK(hipSetDevice(0));
  // Release resources
  for (int dev = 0; dev < devicecount; dev++) {
    HIP_CHECK(hipMemRelease(handle[dev]));
  }
  HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
}

/**
 * Test Description
 * ------------------------
 *    - Check if a partial part of a VMM range can be mapped/unmapped
 * to a physical address.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemMap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemMap_MapPartialVMMMem") {
  int deviceId = 0;
  size_t granularity = 0;
  size_t buffer_size = N * sizeof(int);
  hipDevice_t device;
  CTX_CREATE();
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;  // Current Devices
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
  hipMemGenericAllocationHandle_t handle;
  // Allocate host memory and intialize data
  std::vector<int> A_h(N), B_h(N);
  // Initialize with data
  for (size_t idx = 0; idx < N; idx++) {
    A_h[idx] = idx;
  }
  // Allocate a bigger physical memory chunk of size_mem
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  // Allocate virtual address range of size twice size_mem
  hipDeviceptr_t ptrA;
  HIP_CHECK(hipMemAddressReserve(&ptrA, 2 * size_mem, 0, 0, 0));
  hipMemAccessDesc accessDesc = {};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = device;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  std::fill(B_h.begin(), B_h.end(), initializer);
  HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle, 0));
  HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));
  HIP_CHECK(hipMemcpyHtoD(ptrA, A_h.data(), buffer_size));
  HIP_CHECK(hipMemcpyDtoH(B_h.data(), ptrA, buffer_size));
  REQUIRE(true == std::equal(B_h.begin(), B_h.end(), A_h.data()));
  HIP_CHECK(hipMemUnmap(ptrA, size_mem));
  // Release resources
  HIP_CHECK(hipMemRelease(handle));
  HIP_CHECK(hipMemAddressFree(ptrA, 2 * size_mem));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Negative Argument Tests
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemMap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemMap_negative") {
  size_t granularity = 0;
  size_t buffer_size = N * sizeof(int);
  CTX_CREATE();
  int deviceId = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;  // Current Devices
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
  hipMemGenericAllocationHandle_t handle;
  hipDeviceptr_t ptrA;
  // Allocate physical memory
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  // Allocate virtual address range
  HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));

  SECTION("nullptr to ptrA") {
    REQUIRE(hipMemMap((hipDeviceptr_t) nullptr, size_mem, 0, handle, 0) == hipErrorInvalidValue);
  }

  SECTION("pass zero to size") {
    REQUIRE(hipMemMap(ptrA, 0, 0, handle, 0) == hipErrorInvalidValue);
  }

  HIP_CHECK(hipMemRelease(handle));
  HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
  CTX_DESTROY();
}

/**
 * End doxygen group VirtualMemoryManagementTest.
 * @}
 */
