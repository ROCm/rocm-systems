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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * @addtogroup hipMemSetAccess hipMemSetAccess
 * @{
 * @ingroup VirtualMemoryManagementTest
 * `hipError_t hipMemSetAccess (void* ptr,
 *                              size_t size,
 *                              const hipMemAccessDesc* desc,
 *                              size_t count)` -
 * Set the access flags for each location specified in desc for the given
 * virtual address range.
 */

#ifdef __linux__
#include <unistd.h>
#include <sys/wait.h>
#endif

#include <hip_test_kernels.hh>
#include <hip_test_common.hh>

#include "hipMallocManagedCommon.hh"
#include "hip_vmm_common.hh"

#define THREADS_PER_BLOCK 512
#define NUM_OF_BUFFERS 3
#define DATA_SIZE (1 << 13)
#define NEW_DATA_SIZE (2 * DATA_SIZE)

constexpr int initializer = 0;

/**
 Kernel to perform Square of input data.
 */
static __global__ void square_kernel(int* Buff) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int temp = Buff[i] * Buff[i];
  Buff[i] = temp;
}

// Simple HIP kernel: read from host-backed memory and write to a device buffer
__global__ void copyFromHostMem(const int* hostMem, int* devOut, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) devOut[i] = hostMem[i];
}

/**
 * Test Description
 * ------------------------
 *    - Create a VM mapped to physical memory. Set the access of the
 * VMM chunk to device 0. Validate that flags = hipMemAccessFlagsProtReadWrite
 * is returned by hipMemGetAccess() when location is set to device 0.
 * Validate that flags = hipMemAccessFlagsProtNone is returned by
 * hipMemGetAccess() when location is set to device 1.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemSetGetAccess.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemSetAccess_SetGet") {
  size_t granularity = 0;
  constexpr int N = DATA_SIZE;
  size_t buffer_size = N * sizeof(int);
  int deviceId = 0;
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
  // Validate using hipMemGetAccess()
  hipMemLocation location;
  location.type = hipMemLocationTypeDevice;
  location.id = device;
  unsigned long long flags = 0;  // NOLINT
  HIP_CHECK(hipMemGetAccess(&flags, &location, ptrA));
  REQUIRE(flags == hipMemAccessFlagsProtReadWrite);
  int devicecount = 0;
  HIP_CHECK(hipGetDeviceCount(&devicecount));
  if (devicecount > 1) {
    flags = 0;
    HIP_CHECK(hipDeviceGet(&device, 1));
    location.type = hipMemLocationTypeDevice;
    location.id = device;
    HIP_CHECK(hipMemGetAccess(&flags, &location, ptrA));
    REQUIRE(flags == hipMemAccessFlagsProtNone);
  }
  HIP_CHECK(hipMemUnmap(ptrA, size_mem));
  HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Mult Device Functional Test: Create a VM mapped to physical memory.
 * Set the access of the VMM chunk to both device 0 and device 1.
 * Validate that flags = hipMemAccessFlagsProtReadWrite is returned by
 * hipMemGetAccess() when location is set to device 0. Validate that
 * flags = hipMemAccessFlagsProtReadWrite is returned by hipMemGetAccess()
 * when location is set to device 1.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemSetGetAccess.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemSetAccess_MultDevSetGet") {
  size_t granularity = 0;
  constexpr int N = DATA_SIZE;
  size_t buffer_size = N * sizeof(int);
  int deviceId = 0, device_count = 0;
  hipDevice_t device0, device1;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count < 2) {
    HipTest::HIP_SKIP_TEST("Need 2 GPUs to run test. Skipping Test..");
    return;
  }

  HIP_CHECK(hipDeviceGet(&device0, deviceId));
  checkVMMSupported(device0);
  HIP_CHECK(hipDeviceGet(&device1, (deviceId + 1)));
  checkVMMSupported(device1);
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device0;  // Current Devices
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
  hipMemAccessDesc accessDesc[2];
  accessDesc[0].location.type = hipMemLocationTypeDevice;
  accessDesc[0].location.id = device0;
  accessDesc[0].flags = hipMemAccessFlagsProtReadWrite;
  accessDesc[1].location.type = hipMemLocationTypeDevice;
  accessDesc[1].location.id = device1;
  accessDesc[1].flags = hipMemAccessFlagsProtReadWrite;
  // Make the address accessible to GPU 0 and 1
  HIP_CHECK(hipMemSetAccess(ptrA, size_mem, accessDesc, 2));
  // Validate using hipMemGetAccess()
  hipMemLocation location;
  location.type = hipMemLocationTypeDevice;
  location.id = device0;
  unsigned long long flags = 0;  // NOLINT
  HIP_CHECK(hipMemGetAccess(&flags, &location, ptrA));
  REQUIRE(flags == hipMemAccessFlagsProtReadWrite);
  location.type = hipMemLocationTypeDevice;
  location.id = device1;
  flags = 0;
  HIP_CHECK(hipMemGetAccess(&flags, &location, ptrA));
  REQUIRE(flags == hipMemAccessFlagsProtReadWrite);
  HIP_CHECK(hipMemUnmap(ptrA, size_mem));
  HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
}

/**
 * Test Description
 * ------------------------
 *    - Create a VM mapped to physical memory. Set the access of the VMM chunk
 * to device 0. Validate that flags = 3 is returned by hipMemGetAccess()
 * for entire virtual address range when location is set to device 0.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemSetGetAccess.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemSetAccess_EntireVMMRangeSetGet") {
  size_t granularity = 0;
  constexpr int N = DATA_SIZE;
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
  // Allocate physical memory
  hipMemGenericAllocationHandle_t handle;
  void* ptrA;
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  // Allocate virtual address range
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
  // Validate hipMemGetAccess()
  hipMemLocation location;
  location.type = hipMemLocationTypeDevice;
  location.id = device;
  unsigned long long flags = 0;  // NOLINT
  HIP_CHECK(hipMemGetAccess(&flags, &location, ptrA));
  REQUIRE(flags == hipMemAccessFlagsProtReadWrite);
  unsigned long long uiptr = reinterpret_cast<unsigned long long>(ptrA);
  uiptr += (size_mem - 1);
  HIP_CHECK(hipMemGetAccess(&flags, &location, reinterpret_cast<void*>(uiptr)));
  REQUIRE(flags == hipMemAccessFlagsProtReadWrite);
  HIP_CHECK(hipMemUnmap(ptrA, size_mem));
  HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Negative Tests
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemSetGetAccess.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemGetAccess_NegTst") {
  size_t granularity = 0;
  constexpr int N = DATA_SIZE;
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
  // Allocate physical memory
  hipMemGenericAllocationHandle_t handle;
  void* ptrA;
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  // Allocate virtual address range
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
  // Validate hipMemGetAccess() -ve scenarios
  hipMemLocation location;
  location.type = hipMemLocationTypeDevice;
  location.id = device;
  unsigned long long flags = 0;  // NOLINT
  hipError_t status = hipSuccess;
  status = hipMemGetAccess(nullptr, &location, ptrA);
  REQUIRE(status == hipErrorInvalidValue);
  status = hipMemGetAccess(&flags, nullptr, ptrA);
  REQUIRE(status == hipErrorInvalidValue);
  unsigned long long uiptr = reinterpret_cast<unsigned long long>(ptrA);
  uiptr += size_mem;
  status = hipMemGetAccess(&flags, &location, reinterpret_cast<void*>(uiptr));
  REQUIRE(status == hipErrorInvalidValue);
  HIP_CHECK(hipMemUnmap(ptrA, size_mem));
  HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Test VMM functionality on multiple device. In each device, create
 * a VM mapped to physical memory of the device, copy test data to the VM
 * address range, launch a kernel to perform operation on the data and
 * validate the result.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemSetGetAccess.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemSetAccess_FuncTstOnMultDev") {
  size_t granularity = 0;
  constexpr int N = DATA_SIZE;
  size_t buffer_size = N * sizeof(int);
  CTX_CREATE();
  int deviceId = 0, devicecount = 0;
  HIP_CHECK(hipGetDeviceCount(&devicecount));
  if (devicecount < 2) {
    HipTest::HIP_SKIP_TEST("Machine is Single GPU. Skipping Test..");
    return;
  }
  for (deviceId = 0; deviceId < devicecount; deviceId++) {
    HIP_CHECK(hipSetDevice(deviceId));
    checkVMMSupported(deviceId);
    hipMemAllocationProp prop{};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = deviceId;  // Current Devices
    HIP_CHECK(
        hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
    REQUIRE(granularity > 0);
    size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
    // Allocate physical memory
    void* ptrA;
    hipMemGenericAllocationHandle_t handle;
    HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
    // Allocate virtual address range
    HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));
    HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle, 0));
    HIP_CHECK(hipMemRelease(handle));
    // Set access
    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type = hipMemLocationTypeDevice;
    accessDesc.location.id = deviceId;
    accessDesc.flags = hipMemAccessFlagsProtReadWrite;
    // Make the address accessible to GPU deviceId
    std::vector<int> A_h(N), B_h(N);
    // Set Device, for kernel launch to be launched in the right device.
    HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));
    for (int idx = 0; idx < N; idx++) {
      A_h[idx] = idx;
    }
    HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptrA), A_h.data(), buffer_size));
    // Set the A_h to verify with square kernel.
    for (int idx = 0; idx < N; idx++) {
      A_h[idx] = idx * idx;
    }
    // Launch square kernel
    hipLaunchKernelGGL(square_kernel, dim3(N / THREADS_PER_BLOCK), dim3(THREADS_PER_BLOCK), 0, 0,
                       reinterpret_cast<int*>(ptrA));
    HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptrA), buffer_size));
    HIP_CHECK(hipDeviceSynchronize());
    REQUIRE(true == std::equal(B_h.begin(), B_h.end(), A_h.data()));
    HIP_CHECK(hipMemUnmap(ptrA, size_mem));
    HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
  }
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Create a virtual memnory chunk and set the property of
 * the range to read/write. Write to the memory chunk. Change
 * the property of the range to read only. Check if the memory
 * range can be read.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemMap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemSetAccess_ChangeAccessProp") {
  size_t granularity = 0;
  constexpr int N = DATA_SIZE;
  size_t buffer_size = N * sizeof(int);
  int dev = 0;
  hipDevice_t device;
  CTX_CREATE();
  HIP_CHECK(hipDeviceGet(&device, dev));
  checkVMMSupported(device);
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;  // Current Devices
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
  hipMemGenericAllocationHandle_t handle;  // Allocate host memory and intialize data
  std::vector<int> A_h(N), B_h(N);
  // Initialize with data
  for (size_t idx = 0; idx < N; idx++) {
    A_h[idx] = idx;
  }
  // Allocate a physical memory chunk
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  // Allocate num_buf virtual address ranges
  void* ptrA;
  HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));
  HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle, 0));
  hipMemAccessDesc accessDesc = {};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = device;

  SECTION("Change Read to ReadWrite") {
    accessDesc.flags = hipMemAccessFlagsProtRead;
    HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));
    // Change property of virtual memory range to read only
    accessDesc.flags = hipMemAccessFlagsProtReadWrite;
    HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));
    HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptrA), A_h.data(), buffer_size));
    HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptrA), buffer_size));
    REQUIRE(true == std::equal(B_h.begin(), B_h.end(), A_h.data()));
  }

  SECTION("Change Inaccessible to ReadWrite") {
    accessDesc.flags = hipMemAccessFlagsProtNone;
    HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));
    // Change property of virtual memory range to read only
    accessDesc.flags = hipMemAccessFlagsProtReadWrite;
    HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));
    HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptrA), A_h.data(), buffer_size));
    HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptrA), buffer_size));
    REQUIRE(true == std::equal(B_h.begin(), B_h.end(), A_h.data()));
  }

#if HT_NVIDIA
  SECTION("Check error while writing on Read-Only memory") {
    accessDesc.flags = hipMemAccessFlagsProtRead;
    HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));
    REQUIRE(hipErrorInvalidValue ==
            hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptrA), A_h.data(), buffer_size));
  }

  SECTION("Check error while writing on inaccessible memory") {
    accessDesc.flags = hipMemAccessFlagsProtNone;
    HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));
    REQUIRE(hipErrorInvalidValue ==
            hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptrA), A_h.data(), buffer_size));
  }
#endif

  HIP_CHECK(hipMemUnmap(ptrA, size_mem));
  // Release resources
  HIP_CHECK(hipMemRelease(handle));
  HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Test Virtual Memory to Unified Memory data transfer. Allocate
 * a Virtual Memory chunk and a Unified Memory chunk. Test if data can
 * be exchanged between these chunks.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemSetGetAccess.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemSetAccess_Vmm2UnifiedMemCpy") {
  CTX_CREATE();
  auto managed = HmmAttrPrint();
  if (managed != 1) {
    HipTest::HIP_SKIP_TEST("GPU doesn't support managed memory.Skipping Test..");
    return;
  }
  size_t granularity = 0;
  constexpr int N = DATA_SIZE;
  size_t buffer_size = N * sizeof(int);
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
  // Allocate physical memory
  hipMemGenericAllocationHandle_t handle;
  void *ptrA, *ptrB;
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  // Allocate virtual address range
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
  int *ptrA_h, *ptrB_h;
  HIP_CHECK(hipMallocManaged(&ptrA_h, buffer_size));
  HIP_CHECK(hipMallocManaged(&ptrB_h, buffer_size));
  for (int idx = 0; idx < N; idx++) {
    ptrA_h[idx] = idx;
  }
  HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptrA), ptrA_h, buffer_size));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&ptrB), buffer_size));
  HIP_CHECK(hipMemcpyDtoD(reinterpret_cast<hipDeviceptr_t>(ptrB),
                         reinterpret_cast<hipDeviceptr_t>(ptrA), buffer_size));
  HIP_CHECK(hipMemcpyDtoH(ptrB_h, reinterpret_cast<hipDeviceptr_t>(ptrB), buffer_size));
  bool bPassed = true;
  for (int idx = 0; idx < N; idx++) {
    if (ptrB_h[idx] != idx) {
      bPassed = false;
      break;
    }
  }
  REQUIRE(bPassed == true);
  HIP_CHECK(hipFree(reinterpret_cast<void*>(ptrB)));
  HIP_CHECK(hipFree(reinterpret_cast<void*>(ptrA_h)));
  HIP_CHECK(hipFree(reinterpret_cast<void*>(ptrB_h)));
  HIP_CHECK(hipMemUnmap(ptrA, size_mem));
  HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Test Virtual Memory to Device Memory data transfer. Allocate a Virtual
 * Memory chunk and a Device Memory chunk. Test if data can be exchanged
 * between these chunks.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemSetGetAccess.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemSetAccess_Vmm2DevMemCpy") {
  size_t granularity = 0;
  constexpr int N = DATA_SIZE;
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
  // Allocate physical memory
  hipMemGenericAllocationHandle_t handle;
  void *ptrA, *ptrB;
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  // Allocate virtual address range
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
  std::vector<int> A_h(N), B_h(N);
  for (int idx = 0; idx < N; idx++) {
    A_h[idx] = idx;
  }
  HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptrA), A_h.data(), buffer_size));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&ptrB), buffer_size));
  HIP_CHECK(hipMemcpyDtoD(reinterpret_cast<hipDeviceptr_t>(ptrB),
                          reinterpret_cast<hipDeviceptr_t>(ptrA), buffer_size));
  HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptrB), buffer_size));
  REQUIRE(true == std::equal(B_h.begin(), B_h.end(), A_h.data()));
  HIP_CHECK(hipFree(reinterpret_cast<void*>(ptrB)));
  HIP_CHECK(hipMemUnmap(ptrA, size_mem));
  HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - VM to Device Memory Copy. Allocate a Virtual Memory chunk and a
 * Peer Device Memory chunk. Test if data can be exchanged between
 * these chunks using hipMemcpyDtoD.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemSetGetAccess.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemSetAccess_Vmm2PeerDevMemCpy") {
  size_t granularity = 0;
  constexpr int N = DATA_SIZE;
  size_t buffer_size = N * sizeof(int);
  CTX_CREATE();
  int devicecount = 0;
  HIP_CHECK(hipGetDeviceCount(&devicecount));
  if (devicecount < 2) {
    HipTest::HIP_SKIP_TEST("Machine is Single GPU. Skipping Test..");
    return;
  }
  int deviceId = 0, value = 0;
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
  // Allocate physical memory
  hipMemGenericAllocationHandle_t handle;
  void* ptrA;
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  // Allocate virtual address range
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
  std::vector<int> A_h(N), B_h(N);
  for (int idx = 0; idx < N; idx++) {
    A_h[idx] = idx;
  }
  HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptrA), A_h.data(), buffer_size));
  // Check Peer Access
  for (deviceId = 1; deviceId < devicecount; deviceId++) {
    int canAccessPeer = 0;
    hipDevice_t device_other;
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, 0, deviceId));
    if (0 == canAccessPeer) {
      WARN("Machine does not support Peer Access\n");
      break;
    }
    HIP_CHECK(hipDeviceGet(&device_other, deviceId));
    HIP_CHECK(hipDeviceGetAttribute(&value, hipDeviceAttributeVirtualMemoryManagementSupported,
                                    device_other));
    if (value == 0) {
      // Virtual Memory Mgmt is not supported
      WARN("Machine does not support Virtual Memory Management\n");
      break;
    }
    HIP_CHECK(hipSetDevice(deviceId));
    hipMemAccessDesc access = {};
    access.location.type = hipMemLocationTypeDevice;
    access.location.id = deviceId;
    access.flags = hipMemAccessFlagsProtReadWrite;
    // Make the address accessible to GPU 0
    HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &access, 1));
    hipDeviceptr_t dptr_peer;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dptr_peer), buffer_size));
    HIP_CHECK(hipMemcpyDtoD(dptr_peer, reinterpret_cast<hipDeviceptr_t>(ptrA), buffer_size));
    HIP_CHECK(hipMemcpyDtoH(B_h.data(), dptr_peer, buffer_size));
    REQUIRE(true == std::equal(B_h.begin(), B_h.end(), A_h.data()));
    HIP_CHECK(hipFree(reinterpret_cast<void*>(dptr_peer)));
  }
  HIP_CHECK(hipMemUnmap(ptrA, size_mem));
  HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - VM to Device Memory Copy: Allocate a Virtual Memory chunk and
 * a Peer Device Memory chunk. Test if data can be exchanged between
 * these chunks using hipMemcpyPeer.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemSetGetAccess.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemSetAccess_Vmm2PeerPeerMemCpy") {
  size_t granularity = 0;
  constexpr int N = DATA_SIZE;
  size_t buffer_size = N * sizeof(int);
  CTX_CREATE();
  int devicecount = 0;
  HIP_CHECK(hipGetDeviceCount(&devicecount));
  if (devicecount < 2) {
    HipTest::HIP_SKIP_TEST("Machine is Single GPU. Skipping Test..");
    return;
  }
  int deviceId = 0, value = 0;
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
  // Allocate physical memory
  hipMemGenericAllocationHandle_t handle;
  void* ptrA;
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  // Allocate virtual address range
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
  std::vector<int> A_h(N), B_h(N);
  for (int idx = 0; idx < N; idx++) {
    A_h[idx] = idx;
  }
  HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptrA), A_h.data(), buffer_size));
  // Check Peer Access
  for (deviceId = 1; deviceId < devicecount; deviceId++) {
    std::fill(B_h.begin(), B_h.end(), initializer);
    int canAccessPeer = 0;
    hipDevice_t device_other;
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, 0, deviceId));
    if (0 == canAccessPeer) {
      WARN("Machine does not support Peer Access\n");
      break;
    }

    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type = hipMemLocationTypeDevice;
    accessDesc.location.id = deviceId;
    accessDesc.flags = hipMemAccessFlagsProtRead;
    // Make the address accessible to the rest of the GPUs
    HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));

    HIP_CHECK(hipDeviceGet(&device_other, deviceId));
    HIP_CHECK(hipDeviceGetAttribute(&value, hipDeviceAttributeVirtualMemoryManagementSupported,
                                    device_other));
    if (value == 0) {
      // Virtual Memory Mgmt is not supported
      WARN("Machine does not support Virtual Memory Management\n");
      break;
    }
    HIP_CHECK(hipSetDevice(deviceId));
    hipDeviceptr_t dptr_peer;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dptr_peer), buffer_size));
    HIP_CHECK(hipMemcpyPeer(reinterpret_cast<void*>(dptr_peer), deviceId,
                            reinterpret_cast<void*>(ptrA), 0, buffer_size));
    HIP_CHECK(hipMemcpyDtoH(B_h.data(), dptr_peer, buffer_size));
    REQUIRE(true == std::equal(B_h.begin(), B_h.end(), A_h.data()));
    HIP_CHECK(hipFree(reinterpret_cast<void*>(dptr_peer)));
  }
  HIP_CHECK(hipMemUnmap(ptrA, size_mem));
  HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - VM to VM copy: Allocate memory and map it to an address space in
 * device 0(PtrA). Allocate another chunk of memory and map it to an
 * address space in device 0(PtrB). Check if data can be copied from
 * PtrA -> PtrB using hipMemcpy.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemSetGetAccess.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemSetAccess_Vmm2VMMMemCpy") {
  size_t granularity = 0;
  constexpr int N = DATA_SIZE;
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
  // Allocate physical memory
  hipMemGenericAllocationHandle_t handle1, handle2;
  HIP_CHECK(hipMemCreate(&handle1, size_mem, &prop, 0));
  HIP_CHECK(hipMemCreate(&handle2, size_mem, &prop, 0));
  // Allocate virtual address range
  void *ptrA, *ptrB;
  HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));
  HIP_CHECK(hipMemAddressReserve(&ptrB, size_mem, 0, 0, 0));
  HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle1, 0));
  HIP_CHECK(hipMemMap(ptrB, size_mem, 0, handle2, 0));
  HIP_CHECK(hipMemRelease(handle1));
  HIP_CHECK(hipMemRelease(handle2));
  // Set access
  hipMemAccessDesc accessDesc = {};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = device;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  // Make the addresses accessible to GPU 0
  HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));
  HIP_CHECK(hipMemSetAccess(ptrB, size_mem, &accessDesc, 1));
  std::vector<int> A_h(N), B_h(N);
  for (int idx = 0; idx < N; idx++) {
    A_h[idx] = idx;
  }
  HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptrA), A_h.data(), buffer_size));
  HIP_CHECK(hipMemcpyDtoD(reinterpret_cast<hipDeviceptr_t>(ptrB),
                          reinterpret_cast<hipDeviceptr_t>(ptrA), buffer_size));
  HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptrB), buffer_size));
  REQUIRE(true == std::equal(B_h.begin(), B_h.end(), A_h.data()));
  HIP_CHECK(hipMemUnmap(ptrA, size_mem));
  HIP_CHECK(hipMemUnmap(ptrB, size_mem));
  HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
  HIP_CHECK(hipMemAddressFree(ptrB, size_mem));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Functional Test: Allocate memory and map it to an address space in
 * device 0(PtrA). Allocate another chunk of memory and map it to an
 * address space in device 1(PtrB). Check if data can be copied from
 * PtrA -> PtrB using hipMemcpyPeer.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemSetGetAccess.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemSetAccess_Vmm2VMMInterDevMemCpy") {
  size_t granularity = 0;
  constexpr int N = DATA_SIZE;
  size_t buffer_size = N * sizeof(int);
  CTX_CREATE();
  int devicecount = 0;
  HIP_CHECK(hipGetDeviceCount(&devicecount));
  if (devicecount < 2) {
    HipTest::HIP_SKIP_TEST("Machine is Single GPU. Skipping Test..");
    return;
  }
  int deviceId = 0, value = 0;
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
  std::vector<int> A_h(N), B_h(N);
  for (int idx = 0; idx < N; idx++) {
    A_h[idx] = idx;
  }
  HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptrA), A_h.data(), buffer_size));
  for (deviceId = 1; deviceId < devicecount; deviceId++) {
    int canAccessPeer = 0;
    hipDevice_t device_other;
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, 0, deviceId));
    if (0 == canAccessPeer) {
      WARN("Machine does not support Peer Access\n");
      break;
    }
    std::fill(B_h.begin(), B_h.end(), initializer);
    HIP_CHECK(hipDeviceGet(&device_other, deviceId));
    HIP_CHECK(hipDeviceGetAttribute(&value, hipDeviceAttributeVirtualMemoryManagementSupported,
                                    device_other));
    if (value == 0) {
      // Virtual Memory Mgmt is not supported
      WARN("Machine does not support Virtual Memory Management\n");
      break;
    }
    HIP_CHECK(hipSetDevice(deviceId));
    hipMemAllocationProp prop_loc{};
    prop_loc.type = hipMemAllocationTypePinned;
    prop_loc.location.type = hipMemLocationTypeDevice;
    prop_loc.location.id = device_other;  // Current Devices
    HIP_CHECK(hipMemGetAllocationGranularity(&granularity, &prop_loc,
                                             hipMemAllocationGranularityMinimum));
    size_t size_mem_loc = ((granularity + buffer_size - 1) / granularity) * granularity;
    hipMemGenericAllocationHandle_t handle_loc;
    // Allocate physical memory
    HIP_CHECK(hipMemCreate(&handle_loc, size_mem_loc, &prop_loc, 0));
    // Allocate virtual address range
    void* ptrB;
    HIP_CHECK(hipMemAddressReserve(&ptrB, size_mem_loc, 0, 0, 0));
    HIP_CHECK(hipMemMap(ptrB, size_mem_loc, 0, handle_loc, 0));
    HIP_CHECK(hipMemRelease(handle_loc));
    // Set access
    hipMemAccessDesc accessDesc_loc = {};
    accessDesc_loc.location.type = hipMemLocationTypeDevice;
    accessDesc_loc.location.id = device_other;
    accessDesc_loc.flags = hipMemAccessFlagsProtReadWrite;
    // Make the address accessible to GPU 0
    HIP_CHECK(hipMemSetAccess(ptrB, size_mem_loc, &accessDesc_loc, 1));
    HIP_CHECK(hipMemcpyPeer(ptrB, deviceId, ptrA, 0, buffer_size));
    HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptrB), buffer_size));
    REQUIRE(true == std::equal(B_h.begin(), B_h.end(), A_h.data()));
    HIP_CHECK(hipMemUnmap(ptrB, size_mem_loc));
    HIP_CHECK(hipMemAddressFree(ptrB, size_mem_loc));
  }
  HIP_CHECK(hipMemUnmap(ptrA, size_mem));
  HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
  CTX_DESTROY();
}

class vmm_resize_class {
  size_t current_size_tot;
  size_t current_size_rounded_tot;
  void* ptrVmm;
  std::vector<hipMemGenericAllocationHandle_t> vhandle;
  std::vector<size_t> vsize;
  // allocate initial VMM memory chunk
  int allocate_vmm(hipDeviceptr_t* ptr, hipDevice_t device, size_t size) {
    size_t granularity = 0;
    hipMemAllocationProp prop{};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = device;  // Current Devices
    HIP_CHECK(
        hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
    REQUIRE(granularity > 0);
    size_t size_rounded = ((granularity + size - 1) / granularity) * granularity;
    hipMemGenericAllocationHandle_t handle;
    // Allocate physical memory
    HIP_CHECK(hipMemCreate(&handle, size_rounded, &prop, 0));
    // Store the handle for future reference
    vhandle.push_back(handle);
    vsize.push_back(size_rounded);
    // Allocate virtual address range
    HIP_CHECK(hipMemAddressReserve(&ptrVmm, size_rounded, 0, 0, 0));
    HIP_CHECK(hipMemMap(ptrVmm, size_rounded, 0, handle, 0));
    // Set access
    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type = hipMemLocationTypeDevice;
    accessDesc.location.id = device;
    accessDesc.flags = hipMemAccessFlagsProtReadWrite;
    // Make the address accessible to GPU device
    HIP_CHECK(hipMemSetAccess(ptrVmm, size_rounded, &accessDesc, 1));
    *ptr = reinterpret_cast<hipDeviceptr_t>(ptrVmm);
    current_size_tot += size;
    current_size_rounded_tot += size_rounded;
    return 0;
  }

 public:
  vmm_resize_class(hipDeviceptr_t* ptr, hipDevice_t device, size_t size)
      : current_size_tot(0), current_size_rounded_tot(0) {
    allocate_vmm(ptr, device, size);
  }
  // Free all VMM
  void free_vmm() {
    for (hipMemGenericAllocationHandle_t& myhandle : vhandle) {
      HIP_CHECK(hipMemRelease(myhandle));
    }
    HIP_CHECK(hipMemUnmap(ptrVmm, current_size_rounded_tot));
    HIP_CHECK(hipMemAddressFree(ptrVmm, current_size_rounded_tot));
  }
  // grow memory chunk
  int grow_vmm(hipDeviceptr_t* ptr, hipDevice_t device, size_t size) {
    size_t granularity = 0;
    if (size <= current_size_tot) {
      return -1;
    }
    hipMemAllocationProp prop{};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = device;  // Current Devices
    HIP_CHECK(
        hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
    REQUIRE(granularity > 0);
    // diff size
    size_t grow_size = (size - current_size_tot);
    size_t size_rounded = ((granularity + grow_size - 1) / granularity) * granularity;
    hipMemGenericAllocationHandle_t handle;
    // Allocate physical memory
    HIP_CHECK(hipMemCreate(&handle, size_rounded, &prop, 0));
    // Store the handle for future reference
    vhandle.push_back(handle);
    vsize.push_back(size_rounded);
    // Allocate virtual address range
    // Unmap and Free the old vmm
    HIP_CHECK(hipMemUnmap(ptrVmm, current_size_rounded_tot));
    HIP_CHECK(hipMemAddressFree(ptrVmm, current_size_rounded_tot));
    HIP_CHECK(hipMemAddressReserve(&ptrVmm, (size_rounded + current_size_rounded_tot), 0, 0, 0));
    int idx = 0;
    for (hipMemGenericAllocationHandle_t& myhandle : vhandle) {
      if (idx == 0) {
        HIP_CHECK(hipMemMap(ptrVmm, vsize[idx], 0, myhandle, 0));
      } else {
        unsigned long long uiptr = reinterpret_cast<unsigned long long>(ptrVmm);
        uiptr = uiptr + vsize[idx - 1];
        HIP_CHECK(hipMemMap(reinterpret_cast<void*>(uiptr), vsize[idx], 0, myhandle, 0));
      }
      idx++;
    }
    // Set access
    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type = hipMemLocationTypeDevice;
    accessDesc.location.id = device;
    accessDesc.flags = hipMemAccessFlagsProtReadWrite;
    // Make the address accessible to GPU 0
    HIP_CHECK(hipMemSetAccess(ptrVmm, (size_rounded + current_size_rounded_tot), &accessDesc, 1));
    *ptr = reinterpret_cast<hipDeviceptr_t>(ptrVmm);
    current_size_tot += size;
    current_size_rounded_tot += size_rounded;
    return 0;
  }
};

/**
 * Test Description
 * ------------------------
 *    - Testing memory resize: Allocate physical memory and map it to virtual
 * address range (PtrA). After setting device permission, copy data from
 * host to device. Allocate another chunk of memory of a different size.
 * Map the new chunk to offset (PtrA + size of old chunk).
 * After setting device permission, copy data from host to device at
 * offset (PtrA + size of old chunk). Validate both the old data and new
 * data after copying back to host.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemSetGetAccess.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemSetAccess_GrowVMM") {
  hipDeviceptr_t ptr;
  constexpr int N = DATA_SIZE;
  size_t buffer_size = N * sizeof(int);
  int deviceId = 0;
  hipDevice_t device;
  CTX_CREATE();
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  // Create VMM Object of size buffer_size
  vmm_resize_class resizeobj(&ptr, device, buffer_size);
  // Inititalize Host Buffer
  int* ptrA_h = static_cast<int*>(malloc(buffer_size));
  REQUIRE(ptrA_h != nullptr);
  for (int idx = 0; idx < N; idx++) {
    ptrA_h[idx] = idx;
  }
  // Copy to VMM
  HIP_CHECK(hipMemcpyHtoD(ptr, ptrA_h, buffer_size));
  // Resize the VMM
  int Nnew = NEW_DATA_SIZE;
  size_t buffer_size_new = Nnew * sizeof(int);
  if (-1 == resizeobj.grow_vmm(&ptr, device, buffer_size_new)) {
    WARN("Virtual Memory Management Grow Failed");
    return;
  }
  free(ptrA_h);
  ptrA_h = static_cast<int*>(malloc(buffer_size_new - buffer_size));
  REQUIRE(ptrA_h != nullptr);
  for (int idx = 0; idx < (Nnew - N); idx++) {
    ptrA_h[idx] = N + idx;
  }
  int* ptrB_h = static_cast<int*>(malloc(buffer_size_new));
  REQUIRE(ptrB_h != nullptr);
  unsigned long long uiptr = reinterpret_cast<unsigned long long>(ptr);
  uiptr = uiptr + buffer_size;
  HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(uiptr), ptrA_h,
                          (buffer_size_new - buffer_size)));
  HIP_CHECK(hipMemcpyDtoH(ptrB_h, ptr, buffer_size_new));
  bool bPassed = true;
  for (int idx = 0; idx < Nnew; idx++) {
    if (ptrB_h[idx] != idx) {
      bPassed = false;
      break;
    }
  }
  REQUIRE(bPassed == true);
  free(ptrB_h);
  free(ptrA_h);
  resizeobj.free_vmm();
  CTX_DESTROY();
}

std::atomic<int> bTestPassed{1};
#define NUM_THREADS 5
void test_thread(hipDevice_t device) {
  hipDeviceptr_t ptr;
  constexpr int N = DATA_SIZE;
  size_t buffer_size = N * sizeof(int);
  // Create VMM Object of size buffer_size
  vmm_resize_class vmmobj(&ptr, device, buffer_size);
  // Inititalize Host Buffer
  int* ptrA_h = static_cast<int*>(malloc(buffer_size));
  REQUIRE(ptrA_h != nullptr);
  for (int idx = 0; idx < N; idx++) {
    ptrA_h[idx] = idx;
  }
  // Copy to VMM
  CTX_CREATE();
  HIP_CHECK(hipMemcpyHtoD(ptr, ptrA_h, buffer_size));
  int* ptrB_h = static_cast<int*>(malloc(buffer_size));
  REQUIRE(ptrB_h != nullptr);
  HIP_CHECK(hipMemcpyDtoH(ptrB_h, ptr, buffer_size));
  bool bPassed = true;
  for (int idx = 0; idx < N; idx++) {
    if (ptrB_h[idx] != idx) {
      bPassed = false;
      break;
    }
  }
  if (bPassed) {
    bTestPassed.fetch_and(1);
  } else {
    bTestPassed.fetch_and(0);
  }
  free(ptrB_h);
  free(ptrA_h);
  vmmobj.free_vmm();
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Multithreaded test: Allocate unique virtual memory chunks from
 * multiple threads. Transfer data to these chunks from host and execute
 * kernel function on these data. Validate the results.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemSetGetAccess.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemSetAccess_Multithreaded") {
  CTX_CREATE();
  int deviceId = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);
  std::thread T[NUM_THREADS];
  for (int i = 0; i < NUM_THREADS; i++) {
    T[i] = std::thread(test_thread, device);
  }
  // Wait until all the threads finish their execution
  for (int i = 0; i < NUM_THREADS; i++) {
    T[i].join();
  }
  REQUIRE(1 == bTestPassed.load());
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Negative Tests for hipMemSetAccess()
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemSetGetAccess.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemSetAccess_negative") {
  size_t granularity = 0;
  constexpr int N = DATA_SIZE;
  size_t buffer_size = N * sizeof(int);
  int deviceId = 0;
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

  SECTION("nullptr to ptrA") {
    REQUIRE(hipMemSetAccess(nullptr, size_mem, &accessDesc, 1) == hipErrorInvalidValue);
  }

  SECTION("pass zero to size") {
    REQUIRE(hipMemSetAccess(ptrA, 0, &accessDesc, 1) == hipErrorInvalidValue);
  }

  SECTION("pass a size greater than reserved size") {
    REQUIRE(hipMemSetAccess(ptrA, size_mem + 1, &accessDesc, 1) == hipErrorInvalidValue);
  }

  SECTION("pass a size less than reserved size") {
    REQUIRE(hipMemSetAccess(ptrA, size_mem - 1, &accessDesc, 1) == hipErrorInvalidValue);
  }

  SECTION("invalid location type") {
    accessDesc.location.type = hipMemLocationTypeInvalid;
    REQUIRE(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1) == hipErrorInvalidValue);
  }

  SECTION("invalid id") {
    accessDesc.location.id = -1;
    REQUIRE(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1) == hipErrorInvalidValue);
  }

  SECTION("pass location id as > highest device number") {
    int numDevices = 0;
    HIP_CHECK(hipGetDeviceCount(&numDevices));
    accessDesc.location.id = numDevices;  // set to non existing device
    REQUIRE(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1) == hipErrorInvalidValue);
  }

  SECTION("invalid flag") {
    accessDesc.flags = static_cast<hipMemAccessFlags>(-1);
    REQUIRE(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1) == hipErrorInvalidValue);
  }

  SECTION(" pass zero to count") {
    REQUIRE(hipMemSetAccess(ptrA, size_mem, &accessDesc, 0) == hipErrorInvalidValue);
  }

  SECTION("pass desc as nullptr") {
    REQUIRE(hipMemSetAccess(ptrA, size_mem, nullptr, 1) == hipErrorInvalidValue);
  }

  SECTION("uninitialized virtual memory") {
    void* ptrB;
    HIP_CHECK(hipMemAddressReserve(&ptrB, size_mem, 0, 0, 0));
    REQUIRE(hipMemSetAccess(ptrB, size_mem, &accessDesc, 1) == hipErrorInvalidValue);
    HIP_CHECK(hipMemAddressFree(ptrB, size_mem));
  }

  HIP_CHECK(hipMemUnmap(ptrA, size_mem));
  SECTION("unmapped virtual memory") {
    REQUIRE(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1) == hipErrorInvalidValue);
  }

  HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
  HIP_CHECK(hipMemRelease(handle));
  CTX_DESTROY();
}

TEST_CASE("Unit_hipMemSetGetAccess_Capture") {
  CTX_CREATE();

  const size_t kBufferBytes = DATA_SIZE * sizeof(int);
  const int kDeviceId = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, kDeviceId));
  checkVMMSupported(device);

  hipMemAllocationProp alloc_prop{};
  alloc_prop.type = hipMemAllocationTypePinned;
  alloc_prop.location.type = hipMemLocationTypeDevice;
  alloc_prop.location.id = device;

  size_t granularity = 0;
  HIP_CHECK(hipMemGetAllocationGranularity(&granularity, &alloc_prop,
                                           hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);

  const size_t vmm_bytes = ((granularity + kBufferBytes - 1) / granularity) * granularity;

  hipMemGenericAllocationHandle_t mem_handle;
  HIP_CHECK(hipMemCreate(&mem_handle, vmm_bytes, &alloc_prop, 0));

  hipDeviceptr_t vmm_ptr = nullptr;
  HIP_CHECK(hipMemAddressReserve(&vmm_ptr, vmm_bytes, 0, 0, 0));
  HIP_CHECK(hipMemMap(vmm_ptr, vmm_bytes, 0, mem_handle, 0));
  HIP_CHECK(hipMemRelease(mem_handle));

  hipMemAccessDesc access_desc{};
  access_desc.location.type = hipMemLocationTypeDevice;
  access_desc.location.id = device;
  access_desc.flags = hipMemAccessFlagsProtReadWrite;

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  GENERATE_CAPTURE();

  // Test hipMemSetAccess inside stream capture
  BEGIN_CAPTURE(stream);
  HIP_CHECK(hipMemSetAccess(vmm_ptr, vmm_bytes, &access_desc, 1));
  END_CAPTURE(stream);

  // Test hipMemGetAccess inside stream capture
  BEGIN_CAPTURE(stream);
  hipMemLocation mem_location{};
  mem_location.type = hipMemLocationTypeDevice;
  mem_location.id = device;
  unsigned long long access_flags = 0;
  HIP_CHECK(hipMemGetAccess(&access_flags, &mem_location, vmm_ptr));
  END_CAPTURE(stream);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipMemUnmap(vmm_ptr, vmm_bytes));
  HIP_CHECK(hipMemAddressFree(vmm_ptr, vmm_bytes));
  CTX_DESTROY();
}

TEST_CASE("Unit_hipMemSetAccessHostDevice_hostalloc") {
  // Ensure device 0 is selected
  REQUIRE(hipSetDevice(0) == hipSuccess);

  // ---- Describe a HOST-backed allocation (NUMA-unaware) ----
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;       // pinned system RAM
  prop.location.type = hipMemLocationTypeHost;  // generic host
  prop.location.id = 0;                         // host id must be 0
  prop.requestedHandleTypes = hipMemHandleTypeNone;

  constexpr size_t N = 1024;
  constexpr size_t bytes = N * sizeof(int);

  // get minimum granularity
  size_t gran = 0;
  HIP_CHECK(hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityMinimum));
  size_t mapSize = ((bytes + gran - 1) / gran) * gran;

  // Create host-backed allocation handle
  hipMemGenericAllocationHandle_t handle{};
  HIP_CHECK(hipMemCreate(&handle, mapSize, &prop, 0 /*flags*/));

  // Reserve VA and map
  void* addr = nullptr;
  HIP_CHECK(hipMemAddressReserve(&addr, mapSize, 0 /*align*/, 0 /*addr*/, 0 /*flags*/));

  HIP_CHECK(hipMemMap(addr, mapSize, 0 /*offset*/, handle, 0 /*flags*/));

  // Grant HOST access so the CPU can touch the VA range
  hipMemAccessDesc accHost{};
  accHost.flags = hipMemAccessFlagsProtReadWrite;
  accHost.location.type = hipMemLocationTypeHost;
  accHost.location.id = 0;
  HIP_CHECK(hipMemSetAccess(addr, mapSize, &accHost, 1));

  // Also grant DEVICE access so GPU can read/write it
  hipMemAccessDesc accDev{};
  accDev.flags = hipMemAccessFlagsProtReadWrite;
  accDev.location.type = hipMemLocationTypeDevice;
  accDev.location.id = 0;

  HIP_CHECK(hipMemSetAccess(addr, mapSize, &accDev, 1));

  // ---- CPU can now safely write to the mapping ----
  int* hostPtr = reinterpret_cast<int*>(addr);
  for (size_t i = 0; i < N; ++i) hostPtr[i] = static_cast<int>(i);

  // Device output buffer
  int* dOut = nullptr;
  HIP_CHECK(hipMalloc(&dOut, bytes));

  // Launch kernel to read host memory and write to device buffer
  dim3 block(256), grid((N + block.x - 1) / block.x);
  hipLaunchKernelGGL(copyFromHostMem, grid, block, 0, 0, reinterpret_cast<const int*>(addr), dOut,
                     static_cast<int>(N));
  HIP_CHECK(hipDeviceSynchronize());

  // Verify
  std::vector<int> out(N, -1);
  HIP_CHECK(hipMemcpy(out.data(), dOut, bytes, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < N; ++i) {
    REQUIRE(out[i] == static_cast<int>(i));
  }

  // Cleanup
  HIP_CHECK(hipFree(dOut));
  HIP_CHECK(hipMemUnmap(addr, mapSize));
  HIP_CHECK(hipMemAddressFree(addr, mapSize));
  HIP_CHECK(hipMemRelease(handle));
}

TEST_CASE("Unit_hipMemSetAccessHost_devicealloc") {
  // Ensure device 0 is selected
  REQUIRE(hipSetDevice(0) == hipSuccess);

  // ---- Describe a DEVICE-backed allocation
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;         // pinned system RAM
  prop.location.type = hipMemLocationTypeDevice;  // generic host
  prop.location.id = 0;                           // host id must be 0
  prop.requestedHandleTypes = hipMemHandleTypeNone;

  constexpr size_t N = 1024;
  constexpr size_t bytes = N * sizeof(int);

  //get minimum granularity
  size_t gran = 0;
  HIP_CHECK(hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityMinimum));
  size_t mapSize = ((bytes + gran - 1) / gran) * gran;

  // Create host-backed allocation handle
  hipMemGenericAllocationHandle_t handle{};
  HIP_CHECK(hipMemCreate(&handle, mapSize, &prop, 0 /*flags*/));

  // Reserve VA and map
  void* addr = nullptr;
  HIP_CHECK(hipMemAddressReserve(&addr, mapSize, 0 /*align*/, 0 /*addr*/, 0 /*flags*/));

  HIP_CHECK(hipMemMap(addr, mapSize, 0 /*offset*/, handle, 0 /*flags*/));

  // Grant HOST access. 
  hipMemAccessDesc accHost{};
  accHost.flags = hipMemAccessFlagsProtReadWrite;
  accHost.location.type = hipMemLocationTypeHost;
  accHost.location.id = 0;
  HIP_CHECK_ERROR(hipMemSetAccess(addr, mapSize, &accHost, 1), hipErrorInvalidValue);

  HIP_CHECK(hipMemUnmap(addr, mapSize));
  HIP_CHECK(hipMemAddressFree(addr, mapSize));
  HIP_CHECK(hipMemRelease(handle));
}

/**
 * End doxygen group VirtualMemoryManagementTest.
 * @}
 */
