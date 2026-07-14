/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

// The graphics interop APIs register and map resources owned by a graphics API
// (OpenGL, Vulkan, D3D). A device-only contract harness cannot create a valid
// hipGraphicsResource_t, so these contracts exercise only the externally
// observable invariant that does not need a valid resource: invalid input (a
// null resource array/handle or a zero/negative count) must be rejected with a
// defined error rather than silently succeeding or crashing. The exact error
// code is backend- and platform-specific, so only a non-success status is
// required.
namespace {
void RequireRejected(hipError_t status) {
  REQUIRE(status != hipSuccess);
  // A rejected call leaves a sticky thread-local error; clear it so it does not
  // leak into later tests.
  (void)hipGetLastError();
}
}  // namespace

HIP_TEST_CASE(Contract_GraphicsInterop_MapResources_NullResources_IsRejected) {
  // Mapping with a null resource array (and zero count) is invalid input and
  // must be rejected rather than silently succeeding.
  RequireRejected(hipGraphicsMapResources(0, nullptr, nullptr));
}

HIP_TEST_CASE(Contract_GraphicsInterop_UnmapResources_NullResources_IsRejected) {
  // Unmapping with a null resource array is invalid input and must be rejected.
  RequireRejected(hipGraphicsUnmapResources(0, nullptr, nullptr));
}

HIP_TEST_CASE(Contract_GraphicsInterop_UnregisterResource_NullHandle_IsRejected) {
  // Unregistering a null resource handle is invalid input and must be rejected
  // rather than silently succeeding.
  RequireRejected(hipGraphicsUnregisterResource(nullptr));
}

HIP_TEST_CASE(Contract_GraphicsInterop_ResourceGetMappedPointer_NullHandle_IsRejected) {
  // Querying the mapped device pointer of a null resource is invalid input and
  // must be rejected rather than returning a pointer.
  void* device_ptr = nullptr;
  size_t size = 0;
  RequireRejected(hipGraphicsResourceGetMappedPointer(&device_ptr, &size, nullptr));
}

HIP_TEST_CASE(Contract_GraphicsInterop_SubResourceGetMappedArray_NullHandle_IsRejected) {
  // Querying the mapped array of a null resource is invalid input and must be
  // rejected rather than returning an array handle.
  hipArray_t array = nullptr;
  RequireRejected(hipGraphicsSubResourceGetMappedArray(&array, nullptr, 0, 0));
}
