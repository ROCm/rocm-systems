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
 * @addtogroup hipMemExportToShareableHandle hipMemExportToShareableHandle
 * @{
 * @ingroup VirtualMemoryManagementTest
 * `hipError_t hipMemExportToShareableHandle(void *shareableHandle,
 *                                           hipMemGenericAllocationHandle_t handle,
 *                                           hipMemAllocationHandleType handleType,
 *                                           unsigned long long flags)` -
 * Exports an allocation to a requested shareable handle type.
 */

#include <hip_test_common.hh>
#include "hip_vmm_common.hh"

/**
 * Test Description
 * ------------------------
 *    - Basic sanity test.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemExportToShareableHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemExportToShareableHandle_Positive_Basic") {
  HIP_CHECK(hipFree(0));

  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);

  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;

  size_t granularity;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));

  hipMemGenericAllocationHandle_t handle;
  HIP_CHECK(hipMemCreate(&handle, granularity * 2, &prop, 0));

  void* shareable_handle = nullptr;
  HIP_CHECK(hipMemExportToShareableHandle(&shareable_handle, handle,
                                          hipMemHandleTypePosixFileDescriptor, 0));
  REQUIRE(shareable_handle != nullptr);

  HIP_CHECK(hipMemRelease(handle));
}

/**
 * Test Description
 * ------------------------
 *    - Negative parameters test.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemExportToShareableHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemExportToShareableHandle_Negative_Parameters") {
  HIP_CHECK(hipFree(0));

  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);

  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;

  size_t granularity;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));

  hipMemGenericAllocationHandle_t handle;
  HIP_CHECK(hipMemCreate(&handle, granularity * 2, &prop, 0));

  void* shareable_handle = nullptr;
  SECTION("shareableHandle == nullptr") {
    HIP_CHECK_ERROR(
        hipMemExportToShareableHandle(nullptr, handle, hipMemHandleTypePosixFileDescriptor, 0),
        hipErrorInvalidValue);
  }

  SECTION("handle == nullptr") {
    HIP_CHECK_ERROR(
        hipMemExportToShareableHandle(&shareable_handle, (hipMemGenericAllocationHandle_t) nullptr,
                                      hipMemHandleTypePosixFileDescriptor, 0),
        hipErrorInvalidValue);
  }

  SECTION("invalid handleType") {
    HIP_CHECK_ERROR(
        hipMemExportToShareableHandle(&shareable_handle, handle, hipMemHandleTypeWin32, 0),
        hipErrorInvalidValue);
  }

  SECTION("non-zero flags") {
    HIP_CHECK_ERROR(hipMemExportToShareableHandle(&shareable_handle, handle,
                                                  hipMemHandleTypePosixFileDescriptor, 1),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipMemRelease(handle));
}

TEST_CASE("Unit_hipMemExportToShareableHandle_Capture") {
  CTX_CREATE();

  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);

  hipMemAllocationProp allocation_prop = {};
  allocation_prop.type = hipMemAllocationTypePinned;
  allocation_prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
  allocation_prop.location.type = hipMemLocationTypeDevice;
  allocation_prop.location.id = device;

  size_t allocation_granularity;
  HIP_CHECK(hipMemGetAllocationGranularity(&allocation_granularity, &allocation_prop,
                                           hipMemAllocationGranularityMinimum));

  hipMemGenericAllocationHandle_t allocation_handle;
  HIP_CHECK(hipMemCreate(&allocation_handle, allocation_granularity * 2, &allocation_prop, 0));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  GENERATE_CAPTURE();
  BEGIN_CAPTURE(stream);

  void* shareable_handle = nullptr;
  HIP_CHECK(hipMemExportToShareableHandle(&shareable_handle, allocation_handle,
                                          hipMemHandleTypePosixFileDescriptor, 0));

  END_CAPTURE(stream);

  HIP_CHECK(hipStreamDestroy(stream));
  REQUIRE(shareable_handle != nullptr);
  HIP_CHECK(hipMemRelease(allocation_handle));

  CTX_DESTROY();
}

/**
 * End doxygen group VirtualMemoryManagementTest.
 * @}
 */
