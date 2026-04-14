/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <hip/hip_gl_interop.h>

#include "gl_interop_common.hh"

TEST_CASE(Unit_hipGraphicsUnmapResources_Negative_Parameters) {
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

  HIP_CHECK(hipGraphicsMapResources(1, &vbo_resource, 0));

  SECTION("count == 0") {
    HIP_CHECK_ERROR(hipGraphicsUnmapResources(0, &vbo_resource, 0), hipErrorInvalidValue);
  }

  SECTION("resources == nullptr") {
    HIP_CHECK_ERROR(hipGraphicsUnmapResources(1, nullptr, 0), hipErrorInvalidValue);
  }

  SECTION("not mapped resource") {
    hipGraphicsResource* not_mapped_resource;
    HIP_CHECK(hipGraphicsGLRegisterBuffer(&not_mapped_resource, vbo, hipGraphicsRegisterFlagsNone));
    HIP_CHECK_ERROR(hipGraphicsUnmapResources(1, &not_mapped_resource, 0), hipErrorNotMapped);
    HIP_CHECK(hipGraphicsUnregisterResource(not_mapped_resource));
  }

  SECTION("invalid stream") {
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK_ERROR(hipGraphicsUnmapResources(1, &vbo_resource, stream),
                    hipErrorContextIsDestroyed);
  }

  HIP_CHECK(hipGraphicsUnmapResources(1, &vbo_resource, 0));

  HIP_CHECK(hipGraphicsUnregisterResource(vbo_resource));
}
