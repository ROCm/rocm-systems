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
 * @addtogroup hipMemRetainAllocationHandle hipMemRetainAllocationHandle
 * @{
 * @ingroup VirtualMemoryManagementTest
 * `hipError_t hipMemRetainAllocationHandle(hipMemGenericAllocationHandle_t* handle,
 *                                          void* addr)` -
 * Returns the allocation handle of the backing memory allocation given the address.
 */

#include <hip_test_kernels.hh>
#include <hip_test_common.hh>

#include "hip_vmm_common.hh"

#define DATA_SIZE (1 << 13)

/**
 * Test Description
 * ------------------------
 *    - Create a VM mapped to physical memory. Input addr to
 * hipMemRetainAllocationHandle and validate the handle.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemRetainAllocationHandle.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemRetainAllocationHandle_SetGet") {
  HIP_CHECK(hipFree(0));
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
  hipMemGenericAllocationHandle_t handle;
  hipDeviceptr_t ptrA;
  // Allocate physical memory
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  // Allocate virtual address range
  HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));
  HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle, 0));
  // Test hipMemRetainAllocationHandle
  hipMemGenericAllocationHandle_t gethandle;
  // Check beginning of VMM ptr
  HIP_CHECK(hipMemRetainAllocationHandle(&gethandle, reinterpret_cast<void*>(ptrA)));
  REQUIRE(gethandle == handle);
  HIP_CHECK(hipMemRelease(handle));
  HIP_CHECK(hipMemUnmap(ptrA, size_mem));
  HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
}

/**
 * Test Description
 * ------------------------
 *    - Negative Tests
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemRetainAllocationHandle.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemRetainAllocationHandle_NegTst") {
  HIP_CHECK(hipFree(0));
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
  hipMemGenericAllocationHandle_t handle;
  hipDeviceptr_t ptrA;
  // Allocate physical memory
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  // Allocate virtual address range
  HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));
  HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle, 0));
  // Test hipMemRetainAllocationHandle
  hipMemGenericAllocationHandle_t gethandle;
  SECTION("nullptr handle") {
    REQUIRE(hipMemRetainAllocationHandle(nullptr, reinterpret_cast<void*>(ptrA)) ==
            hipErrorInvalidValue);
  }
  SECTION("nullptr Vmm ptr") {
    REQUIRE(hipMemRetainAllocationHandle(&gethandle, nullptr) == hipErrorInvalidValue);
  }
  SECTION("not mapped address") {
    hipDeviceptr_t ptrB;
    HIP_CHECK(hipMemAddressReserve(&ptrB, size_mem, 0, 0, 0));
    REQUIRE(hipMemRetainAllocationHandle(&gethandle, reinterpret_cast<void*>(ptrB)) ==
            hipErrorInvalidValue);
    HIP_CHECK(hipMemAddressFree(ptrB, size_mem));
  }
  HIP_CHECK(hipMemRelease(handle));
  HIP_CHECK(hipMemUnmap(ptrA, size_mem));
  SECTION("unmapped address") {
    REQUIRE(hipMemRetainAllocationHandle(&gethandle, reinterpret_cast<void*>(ptrA)) ==
            hipErrorInvalidValue);
  }
  HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
}

/**
* End doxygen group VirtualMemoryManagementTest.
* @}
*/
