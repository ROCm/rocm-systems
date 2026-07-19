/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>
#include <iterator>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

// BACKEND-DIFF: the OpenGL-specific interop entry points (hipGLGetDevices,
// hipGraphicsGLRegisterBuffer/Image) live in hip/hip_gl_interop.h. On AMD that
// header is self-contained. On NVIDIA it includes <cuda_gl_interop.h>, which
// includes <GL/gl.h> and so requires OpenGL development headers to be present.
// A headless CUDA node need not have them (the H100 CI node does not), so
// including this header on NVIDIA breaks the translation-unit compile outright.
// The five portable hipGraphics* rejection contracts in this file only need
// hip_runtime_api.h and stay backend-neutral; the three GL-specific contracts
// below are gated to AMD so NVIDIA never pulls in the GL header chain.
#if HT_AMD
#include <hip/hip_gl_interop.h>
#endif

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

// @asserts: hipGraphicsMapResources - a null resource array or null element with positive count is rejected with a defined error
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

// @asserts: hipGraphicsUnmapResources - a null resource array or null element with positive count is rejected with a defined error
HIP_TEST_CASE(Contract_GraphicsInterop_UnmapResources_NullResources_IsRejected) {
  // A positive count with a null resource array must be rejected (count > 0 so
  // the null-array check is reached rather than short-circuited by count <= 0).
  RequireRejected(hipGraphicsUnmapResources(1, nullptr, nullptr));

  // A non-null array whose single element is null must likewise be rejected.
  hipGraphicsResource_t resources[1] = {nullptr};
  RequireRejected(hipGraphicsUnmapResources(1, resources, nullptr));
}

// @asserts: hipGraphicsUnregisterResource - a null resource handle is rejected with a defined error rather than silently succeeding
HIP_TEST_CASE(Contract_GraphicsInterop_UnregisterResource_NullHandle_IsRejected) {
  // Unregistering a null resource handle is invalid input and must be rejected
  // rather than silently succeeding.
  RequireRejected(hipGraphicsUnregisterResource(nullptr));
}

// @asserts: hipGraphicsResourceGetMappedPointer - querying the mapped pointer of a null resource is rejected with a defined error
HIP_TEST_CASE(Contract_GraphicsInterop_ResourceGetMappedPointer_NullHandle_IsRejected) {
  // Querying the mapped device pointer of a null resource is invalid input and
  // must be rejected rather than returning a pointer.
  void* device_ptr = nullptr;
  size_t size = 0;
  RequireRejected(hipGraphicsResourceGetMappedPointer(&device_ptr, &size, nullptr));
}

// @asserts: hipGraphicsSubResourceGetMappedArray - querying the mapped array of a null resource is rejected with a defined error
HIP_TEST_CASE(Contract_GraphicsInterop_SubResourceGetMappedArray_NullHandle_IsRejected) {
  // Querying the mapped array of a null resource is invalid input and must be
  // rejected rather than returning an array handle.
  hipArray_t array = nullptr;
  RequireRejected(hipGraphicsSubResourceGetMappedArray(&array, nullptr, 0, 0));
}

// The OpenGL-specific interop entry points below live in hip_gl_interop.h. Their
// success paths require a current OpenGL context and real GL buffer/image object
// names, which a device-only contract harness never establishes. Each function
// checks for a current GL context first, so with no context bound they must
// report a defined error rather than crashing or silently succeeding. The exact
// code is backend- and platform-specific (the AMD runtime returns
// hipErrorInvalidValue for "no GL context is current"), so only a non-success
// status is required.
//
// BACKEND-DIFF: AMD-only. These are gated with the same #if HT_AMD as the GL
// header include above, because the NVIDIA hip_gl_interop.h pulls in
// <GL/gl.h> (via <cuda_gl_interop.h>), which a headless CUDA node lacks.
#if HT_AMD
// @asserts: hipGLGetDevices - querying HIP devices with no current GL context is rejected with a defined error
HIP_TEST_CASE(Contract_GraphicsInterop_GLGetDevices_NoGLContext_IsRejected) {
  // Querying the HIP devices for the current GL context with no GL context bound
  // must be rejected. A positive device-count buffer size is passed so the query
  // reaches the no-context check rather than short-circuiting on a zero size.
  unsigned int device_count = 0;
  int devices[8] = {};
  RequireRejected(hipGLGetDevices(&device_count, devices,
                                  static_cast<unsigned int>(std::size(devices)),
                                  hipGLDeviceListAll));
}

// @asserts: hipGraphicsGLRegisterBuffer - registering a GL buffer with no current GL context is rejected with a defined error
HIP_TEST_CASE(Contract_GraphicsInterop_GLRegisterBuffer_NoGLContext_IsRejected) {
  // Registering a GL buffer with no current GL context must be rejected. The
  // buffer name is a bogus non-zero GLuint; the no-context check fires before the
  // name is ever dereferenced against GL, so no real GL object is needed.
  hipGraphicsResource* resource = nullptr;
  RequireRejected(hipGraphicsGLRegisterBuffer(&resource, 1u, hipGraphicsRegisterFlagsNone));
}

// @asserts: hipGraphicsGLRegisterImage - registering a GL image with no current GL context is rejected with a defined error
HIP_TEST_CASE(Contract_GraphicsInterop_GLRegisterImage_NoGLContext_IsRejected) {
  // Registering a GL image with no current GL context must be rejected. As above,
  // the image name and target are bogus but never reach GL because the no-context
  // check rejects first. GL_TEXTURE_2D is 0x0DE1.
  constexpr unsigned int kGlTexture2D = 0x0DE1;
  hipGraphicsResource* resource = nullptr;
  RequireRejected(hipGraphicsGLRegisterImage(&resource, 1u, kGlTexture2D,
                                             hipGraphicsRegisterFlagsNone));
}
#endif  // HT_AMD
