/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <hip/hip_gl_interop.h>

#include "gl_interop_common.hh"

TEST_CASE(Unit_hipGraphicsMapResources_Positive_Basic) {
  GLContextScopeGuard gl_context;

  const int device_count = HipTest::getDeviceCount();
  unsigned int gl_device_count = 0;
  std::vector<int> gl_devices(device_count, -1);

  // Initialize GL interop
  HIP_CHECK(hipGLGetDevices(&gl_device_count, gl_devices.data(), device_count, hipGLDeviceListAll));
  REQUIRE(gl_device_count == 1);
  REQUIRE(gl_devices.at(0) == 0);

  GLBufferObject vbo;
  GLImageObject tex;

  std::array<hipGraphicsResource_t, 2> resources;

  HIP_CHECK(hipGraphicsGLRegisterBuffer(&resources.at(0), vbo, hipGraphicsRegisterFlagsNone));
  HIP_CHECK(hipGraphicsGLRegisterImage(&resources.at(1), tex, GL_TEXTURE_2D,
                                       hipGraphicsRegisterFlagsNone));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipGraphicsMapResources(resources.size(), resources.data(), stream));

  HIP_CHECK(hipGraphicsUnmapResources(resources.size(), resources.data(), stream));

  HIP_CHECK(hipStreamDestroy(stream));

  HIP_CHECK(hipGraphicsUnregisterResource(resources.at(0)));
  HIP_CHECK(hipGraphicsUnregisterResource(resources.at(1)));
}

TEST_CASE(Unit_hipGraphicsMapResources_Negative_Parameters) {
  GLContextScopeGuard gl_context;

  const int device_count = HipTest::getDeviceCount();
  unsigned int gl_device_count = 0;
  std::vector<int> gl_devices(device_count, -1);

  // Initialize GL interop
  HIP_CHECK(hipGLGetDevices(&gl_device_count, gl_devices.data(), device_count, hipGLDeviceListAll));
  REQUIRE(gl_device_count == 1);
  REQUIRE(gl_devices.at(0) == 0);

  GLBufferObject vbo;

  hipGraphicsResource* vbo_resource;

  HIP_CHECK(hipGraphicsGLRegisterBuffer(&vbo_resource, vbo, hipGraphicsRegisterFlagsNone));

  SECTION("count == 0") {
    HIP_CHECK_ERROR(hipGraphicsMapResources(0, &vbo_resource, 0), hipErrorInvalidValue);
  }

  SECTION("resources == nullptr") {
    HIP_CHECK_ERROR(hipGraphicsMapResources(1, nullptr, 0), hipErrorInvalidValue);
  }

  SECTION("unregistered resource") {
    hipGraphicsResource* unregistered_resource;
    HIP_CHECK(
        hipGraphicsGLRegisterBuffer(&unregistered_resource, vbo, hipGraphicsRegisterFlagsNone));
    HIP_CHECK(hipGraphicsUnregisterResource(unregistered_resource));
    HIP_CHECK_ERROR(hipGraphicsMapResources(1, &unregistered_resource, 0), hipErrorInvalidHandle);
  }

  SECTION("already mapped resource") {
    HIP_CHECK(hipGraphicsMapResources(1, &vbo_resource, 0));
    HIP_CHECK_ERROR(hipGraphicsMapResources(1, &vbo_resource, 0), hipErrorAlreadyMapped);
    HIP_CHECK(hipGraphicsUnmapResources(1, &vbo_resource, 0));
  }

  SECTION("invalid stream") {
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK_ERROR(hipGraphicsMapResources(1, &vbo_resource, stream), hipErrorContextIsDestroyed);
  }

  HIP_CHECK(hipGraphicsUnregisterResource(vbo_resource));
}
