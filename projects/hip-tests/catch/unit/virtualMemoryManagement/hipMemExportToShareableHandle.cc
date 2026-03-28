/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
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
HIP_TEST_CASE(Unit_hipMemExportToShareableHandle_Positive_Basic) {
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
HIP_TEST_CASE(Unit_hipMemExportToShareableHandle_Negative_Parameters) {
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

HIP_TEST_CASE(Unit_hipMemExportToShareableHandle_Capture) {
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
