/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <hip/hip_gl_interop.h>

#include "gl_interop_common.hh"

TEST_CASE(Unit_hipGraphicsUnregisterResource_Negative_Parameters) {
  GLContextScopeGuard gl_context;

  const int device_count = HipTest::getDeviceCount();
  unsigned int gl_device_count = 0;
  std::vector<int> gl_devices(device_count, -1);

  // Initialize GL interop
  HIP_CHECK(hipGLGetDevices(&gl_device_count, gl_devices.data(), device_count, hipGLDeviceListAll));
  REQUIRE(gl_device_count == 1);
  REQUIRE(gl_devices.at(0) == 0);

  GLBufferObject vbo;

  SECTION("null resource") {
    hipGraphicsResource* null_resource = nullptr;
    HIP_CHECK_ERROR(hipGraphicsUnregisterResource(null_resource), hipErrorInvalidValue);
  }

    SECTION("already unregistered resource") {
    hipGraphicsResource* unregistered_resource;
    HIP_CHECK(
        hipGraphicsGLRegisterBuffer(&unregistered_resource, vbo, hipGraphicsRegisterFlagsNone));
    HIP_CHECK(hipGraphicsUnregisterResource(unregistered_resource));
    HIP_CHECK_ERROR(hipGraphicsUnregisterResource(unregistered_resource), hipErrorInvalidHandle);
  }

  SECTION("mapped resource") {
    hipGraphicsResource* mapped_resource;
    HIP_CHECK(hipGraphicsGLRegisterBuffer(&mapped_resource, vbo, hipGraphicsRegisterFlagsNone));
    HIP_CHECK(hipGraphicsMapResources(1, &mapped_resource, 0));
    HIP_CHECK_ERROR(hipGraphicsUnregisterResource(mapped_resource), hipErrorAlreadyMapped);
  }
}
