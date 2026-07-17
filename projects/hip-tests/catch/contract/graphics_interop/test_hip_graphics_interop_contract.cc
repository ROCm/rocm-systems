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
  // A positive count with a null resource array is invalid input and must be
  // rejected. count must be > 0 so the null-array check is actually reached: the
  // runtime rejects count <= 0 first, so a zero count would never exercise the
  // null-array path.
  RequireRejected(hipGraphicsMapResources(1, nullptr, nullptr));

  // A non-null array whose single element is null must likewise be rejected.
  hipGraphicsResource_t resources[1] = {nullptr};
  RequireRejected(hipGraphicsMapResources(1, resources, nullptr));
}

HIP_TEST_CASE(Contract_GraphicsInterop_UnmapResources_NullResources_IsRejected) {
  // A positive count with a null resource array must be rejected (count > 0 so
  // the null-array check is reached rather than short-circuited by count <= 0).
  RequireRejected(hipGraphicsUnmapResources(1, nullptr, nullptr));

  // A non-null array whose single element is null must likewise be rejected.
  hipGraphicsResource_t resources[1] = {nullptr};
  RequireRejected(hipGraphicsUnmapResources(1, resources, nullptr));
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
