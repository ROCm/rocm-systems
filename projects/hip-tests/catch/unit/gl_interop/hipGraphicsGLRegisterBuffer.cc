/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <hip/hip_gl_interop.h>

#include "gl_interop_common.hh"

namespace {
constexpr std::array<unsigned int, 3> kFlags{hipGraphicsRegisterFlagsNone,
                                             hipGraphicsRegisterFlagsReadOnly,
                                             hipGraphicsRegisterFlagsWriteDiscard};
}  // anonymous namespace

HIP_TEST_CASE(Unit_hipGraphicsGLRegisterBuffer_Positive_Basic) {
  GLContextScopeGuard gl_context;

  const int device_count = HipTest::getDeviceCount();
  unsigned int gl_device_count = 0;
  std::vector<int> gl_devices(device_count, -1);

  // Initialize GL interop
  HIP_CHECK(hipGLGetDevices(&gl_device_count, gl_devices.data(), device_count, hipGLDeviceListAll));
  REQUIRE(gl_device_count == 1);
  REQUIRE(gl_devices.at(0) == 0);

  const auto flags = GENERATE(from_range(begin(kFlags), end(kFlags)));

  GLBufferObject vbo;

  hipGraphicsResource* vbo_resource;

  HIP_CHECK(hipGraphicsGLRegisterBuffer(&vbo_resource, vbo, flags));

  HIP_CHECK(hipGraphicsUnregisterResource(vbo_resource));
}

HIP_TEST_CASE(Unit_hipGraphicsGLRegisterBuffer_Positive_Register_Twice) {
  GLContextScopeGuard gl_context;

  const int device_count = HipTest::getDeviceCount();
  unsigned int gl_device_count = 0;
  std::vector<int> gl_devices(device_count, -1);

  // Initialize GL interop
  HIP_CHECK(hipGLGetDevices(&gl_device_count, gl_devices.data(), device_count, hipGLDeviceListAll));
  REQUIRE(gl_device_count == 1);
  REQUIRE(gl_devices.at(0) == 0);

  GLBufferObject vbo;

  hipGraphicsResource *vbo_resource_1, *vbo_resource_2;

  HIP_CHECK(hipGraphicsGLRegisterBuffer(&vbo_resource_1, vbo, hipGraphicsRegisterFlagsNone));
  HIP_CHECK(hipGraphicsGLRegisterBuffer(&vbo_resource_2, vbo, hipGraphicsRegisterFlagsNone));

  HIP_CHECK(hipGraphicsUnregisterResource(vbo_resource_1));
  HIP_CHECK(hipGraphicsUnregisterResource(vbo_resource_2));
}

HIP_TEST_CASE(Unit_hipGraphicsGLRegisterBuffer_Negative_Parameters) {
  GLContextScopeGuard gl_context;

  GLBufferObject vbo;

  hipGraphicsResource* vbo_resource;

  SECTION("resource == nullptr") {
    HIP_CHECK_ERROR(hipGraphicsGLRegisterBuffer(nullptr, vbo, hipGraphicsRegisterFlagsNone),
                    hipErrorInvalidValue);
  }

  SECTION("invalid buffer") {
    HIP_CHECK_ERROR(
        hipGraphicsGLRegisterBuffer(&vbo_resource, GLuint{}, hipGraphicsRegisterFlagsNone),
        hipErrorInvalidValue);
  }

  SECTION("invalid flags") {
    HIP_CHECK_ERROR(
        hipGraphicsGLRegisterBuffer(&vbo_resource, vbo, std::numeric_limits<unsigned int>::max()),
        hipErrorInvalidValue);
  }

  SECTION("flags == hipGraphicsRegisterFlagsSurfaceLoadStore") {
    HIP_CHECK_ERROR(
        hipGraphicsGLRegisterBuffer(&vbo_resource, vbo, hipGraphicsRegisterFlagsSurfaceLoadStore),
        hipErrorInvalidValue);
  }

  SECTION("flags == hipGraphicsRegisterFlagsTextureGather") {
    HIP_CHECK_ERROR(
        hipGraphicsGLRegisterBuffer(&vbo_resource, vbo, hipGraphicsRegisterFlagsTextureGather),
        hipErrorInvalidValue);
  }
}
