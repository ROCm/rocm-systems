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
constexpr std::array<hipGLDeviceList, 3> kDeviceLists{
    hipGLDeviceListAll, hipGLDeviceListCurrentFrame, hipGLDeviceListNextFrame};
}  // anonymous namespace

TEST_CASE(Unit_hipGLGetDevices_Positive_Basic) {
  GLContextScopeGuard gl_context;

  const auto device_list = GENERATE(from_range(begin(kDeviceLists), end(kDeviceLists)));

  const int device_count = HipTest::getDeviceCount();

  unsigned int gl_device_count = 0;
  std::vector<int> gl_devices(device_count, -1);

  if (device_list == hipGLDeviceListNextFrame) {
    HIP_CHECK_ERROR(hipGLGetDevices(&gl_device_count, gl_devices.data(), device_count, device_list),
                    hipErrorNotSupported);
    REQUIRE(gl_device_count == 0);
    REQUIRE(gl_devices.at(0) == -1);
  } else {
    HIP_CHECK(hipGLGetDevices(&gl_device_count, gl_devices.data(), device_count, device_list));
    REQUIRE(gl_device_count == 1);
    REQUIRE(gl_devices.at(0) == 0);
  }
}

TEST_CASE(Unit_hipGLGetDevices_Positive_Parameters) {
  GLContextScopeGuard gl_context;

  const int device_count = HipTest::getDeviceCount();

  unsigned int gl_device_count = 0;
  std::vector<int> gl_devices(device_count, -1);

  SECTION("pHipDeviceCount == nullptr") {
    HIP_CHECK_ERROR(hipGLGetDevices(nullptr, gl_devices.data(), device_count, hipGLDeviceListAll),
                    hipErrorInvalidValue);
    REQUIRE(gl_devices.at(0) == -1);
  }

  SECTION("pHipDevices == nullptr") {
    HIP_CHECK_ERROR(hipGLGetDevices(&gl_device_count, nullptr, device_count, hipGLDeviceListAll),
                    hipErrorInvalidValue);
    REQUIRE(gl_device_count == 0);
  }

  SECTION("hipDeviceCount == 0") {
    HIP_CHECK_ERROR(hipGLGetDevices(&gl_device_count, gl_devices.data(), 0, hipGLDeviceListAll),
                    hipErrorInvalidValue);
    REQUIRE(gl_device_count == 0);
    REQUIRE(gl_devices.at(0) == -1);
  }
}

TEST_CASE(Unit_hipGLGetDevices_Negative_Parameters) {
  GLContextScopeGuard gl_context;

  const int device_count = HipTest::getDeviceCount();

  unsigned int gl_device_count = 0;
  std::vector<int> gl_devices(device_count, -1);

  SECTION("invalid deviceList") {
    HIP_CHECK_ERROR(hipGLGetDevices(&gl_device_count, gl_devices.data(), device_count,
                                    static_cast<hipGLDeviceList>(-1)),
                    hipErrorInvalidValue);
    REQUIRE(gl_device_count == 0);
    REQUIRE(gl_devices.at(0) == -1);
  }
}
