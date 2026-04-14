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
constexpr std::array<unsigned int, 5> kFlags{
    hipGraphicsRegisterFlagsNone, hipGraphicsRegisterFlagsReadOnly,
    hipGraphicsRegisterFlagsWriteDiscard, hipGraphicsRegisterFlagsSurfaceLoadStore,
    hipGraphicsRegisterFlagsTextureGather};
}  // anonymous namespace

TEST_CASE(Unit_hipGraphicsGLRegisterImage_Positive_Basic) {
  GLContextScopeGuard gl_context;

  const int device_count = HipTest::getDeviceCount();
  unsigned int gl_device_count = 0;
  std::vector<int> gl_devices(device_count, -1);

  // Initialize GL interop
  HIP_CHECK(hipGLGetDevices(&gl_device_count, gl_devices.data(), device_count, hipGLDeviceListAll));
  REQUIRE(gl_device_count == 1);
  REQUIRE(gl_devices.at(0) == 0);

  const auto flags = GENERATE(from_range(begin(kFlags), end(kFlags)));

  GLImageObject tex;

  hipGraphicsResource* tex_resource;

  HIP_CHECK(hipGraphicsGLRegisterImage(&tex_resource, tex, GL_TEXTURE_2D, flags));

  HIP_CHECK(hipGraphicsUnregisterResource(tex_resource));
}

TEST_CASE(Unit_hipGraphicsGLRegisterImage_Positive_Register_Twice) {
  GLContextScopeGuard gl_context;
  const int device_count = HipTest::getDeviceCount();
  unsigned int gl_device_count = 0;
  std::vector<int> gl_devices(device_count, -1);

  // Initialize GL interop
  HIP_CHECK(hipGLGetDevices(&gl_device_count, gl_devices.data(), device_count, hipGLDeviceListAll));
  REQUIRE(gl_device_count == 1);
  REQUIRE(gl_devices.at(0) == 0);

  GLImageObject tex;

  hipGraphicsResource *tex_resource_1, *tex_resource_2;

  HIP_CHECK(hipGraphicsGLRegisterImage(&tex_resource_1, tex, GL_TEXTURE_2D,
                                       hipGraphicsRegisterFlagsNone));
  HIP_CHECK(hipGraphicsGLRegisterImage(&tex_resource_2, tex, GL_TEXTURE_2D,
                                       hipGraphicsRegisterFlagsNone));

  HIP_CHECK(hipGraphicsUnregisterResource(tex_resource_1));
  HIP_CHECK(hipGraphicsUnregisterResource(tex_resource_2));
}

TEST_CASE(Unit_hipGraphicsGLRegisterImage_Negative_Parameters) {
  GLContextScopeGuard gl_context;

  GLImageObject tex;

  hipGraphicsResource* tex_resource;

  SECTION("resource == nullptr") {
    HIP_CHECK_ERROR(
        hipGraphicsGLRegisterImage(nullptr, tex, GL_TEXTURE_2D, hipGraphicsRegisterFlagsNone),
        hipErrorInvalidValue);
  }

  SECTION("invalid image") {
    HIP_CHECK_ERROR(hipGraphicsGLRegisterImage(&tex_resource, GLuint{}, GL_TEXTURE_2D,
                                               hipGraphicsRegisterFlagsNone),
                    hipErrorInvalidValue);
  }

  SECTION("invalid target") {
    HIP_CHECK_ERROR(
        hipGraphicsGLRegisterImage(&tex_resource, tex, GL_BUFFER, hipGraphicsRegisterFlagsNone),
        hipErrorInvalidValue);
  }

  SECTION("target does not match the object") {
    HIP_CHECK_ERROR(hipGraphicsGLRegisterImage(&tex_resource, tex, GL_RENDERBUFFER,
                                               hipGraphicsRegisterFlagsNone),
                    hipErrorInvalidValue);
  }

  SECTION("invalid flags") {
    HIP_CHECK_ERROR(hipGraphicsGLRegisterImage(&tex_resource, tex, GL_TEXTURE_2D,
                                               std::numeric_limits<unsigned int>::max()),
                    hipErrorInvalidValue);
  }
}
