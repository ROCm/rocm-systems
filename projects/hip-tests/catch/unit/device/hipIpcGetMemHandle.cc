/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstring>

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>

/**
 * @addtogroup hipIpcGetMemHandle hipIpcGetMemHandle
 * @{
 * @ingroup DeviceTest
 * `hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr)` -
 * Gets an interprocess memory handle for an existing device memory allocation.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipIpcMemAccess_ParameterValidation
 */

/**
 * Test Description
 * ------------------------
 *  - Check that unique handles are returned in consecutive calls.
 * Test source
 * ------------------------
 *  - unit/device/hipIpcGetMemHandle.cc
 * Test requirements
 * ------------------------
 *  - Host specific (LINUX)
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipIpcGetMemHandle_Positive_Unique_Handles_Separate_Allocations) {
  void *ptr1, *ptr2;
  hipIpcMemHandle_t handle1, handle2;
  HIP_CHECK(hipMalloc(&ptr1, 1024));
  HIP_CHECK(hipMalloc(&ptr2, 1024));
  HIP_CHECK(hipIpcGetMemHandle(&handle1, ptr1));
  HIP_CHECK(hipIpcGetMemHandle(&handle2, ptr2));

  CHECK(memcmp(&handle1, &handle2, sizeof(handle1)) != 0);

  HIP_CHECK(hipFree(ptr1));
  HIP_CHECK(hipFree(ptr2));
}

/**
 * Test Description
 * ------------------------
 *  - Test if previously freed memory will generate an invalid handle:
 *    -# When memory is freed before getting handle
 *      - Expected output: return `hipErrorInvalidValue
 * Test source
 * ------------------------
 *  - unit/device/hipIpcGetMemHandle.cc
 * Test requirements
 * ------------------------
 *  - Host specific (LINUX)
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipIpcGetMemHandle_Negative_Handle_For_Freed_Memory) {
  void* ptr;
  hipIpcMemHandle_t handle;
  HIP_CHECK(hipMalloc(&ptr, 1024));
  HIP_CHECK(hipFree(ptr));
  HIP_CHECK_ERROR(hipIpcGetMemHandle(&handle, ptr), hipErrorInvalidValue);
}

/**
 * Test Description
 * ------------------------
 *  - Test if out of bounds pointer will generate an error:
 *    -# When the memory pointer is too large
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/device/hipIpcGetMemHandle.cc
 * Test requirements
 * ------------------------
 *  - Host specific (LINUX)
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipIpcGetMemHandle_Negative_Out_Of_Bound_Pointer) {
  int* ptr;
  constexpr size_t n = 1024;
  hipIpcMemHandle_t handle;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(*ptr)));
  HIP_CHECK_ERROR(hipIpcGetMemHandle(&handle, reinterpret_cast<void*>(ptr + n)),
                  hipErrorInvalidValue);
  HIP_CHECK(hipFree(reinterpret_cast<void*>(ptr)));
}

/**
 * End doxygen group DeviceTest.
 * @}
 */
