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

HIP_TEST_CASE(Unit_hipGLGetDevices_Positive_Basic) {
  CHECK_IMAGE_SUPPORT

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

#if defined(__linux__) && HT_AMD
HIP_TEST_CASE(Unit_hipGLGetDevices_Positive_GLXAfterNoCurrentContext) {
  CHECK_IMAGE_SUPPORT

  if (GetGLContextType() != GLContextType::Glut) {
    HIP_SKIP_TEST("GLX recovery requires GL_CONTEXT_TYPE=GLUT");
  }

  GLContextScopeGuard gl_context;
  const int device_count = HipTest::getDeviceCount();
  REQUIRE(device_count > 0);
  unsigned int gl_device_count = 0;
  std::vector<int> gl_devices(device_count, -1);
  HIP_CHECK(hipGLGetDevices(&gl_device_count, gl_devices.data(), device_count, hipGLDeviceListAll));
  REQUIRE(gl_device_count == 1);

  Display* display = glXGetCurrentDisplay();
  const GLXContext context = glXGetCurrentContext();
  const GLXDrawable draw = glXGetCurrentDrawable();
  const GLXDrawable read = glXGetCurrentReadDrawable();
  REQUIRE(display != nullptr);
  REQUIRE(context != nullptr);
  REQUIRE(glXMakeContextCurrent(display, None, None, nullptr));
  const hipError_t no_context_result =
      hipGLGetDevices(&gl_device_count, gl_devices.data(), device_count, hipGLDeviceListAll);
  // Restore before checking the error so an assertion cannot strand the shared GLUT context.
  REQUIRE(glXMakeContextCurrent(display, draw, read, context));
  HIP_CHECK_ERROR(no_context_result, hipErrorInvalidValue);

  // Reusing the same context must repair the display overwritten by failed association.
  gl_device_count = 0;
  std::fill(gl_devices.begin(), gl_devices.end(), -1);
  HIP_CHECK(hipGLGetDevices(&gl_device_count, gl_devices.data(), device_count, hipGLDeviceListAll));
  REQUIRE(gl_device_count == 1);
  REQUIRE(gl_devices.at(0) == 0);
}
#endif

#if defined(USE_EGL) && defined(__linux__) && HT_AMD
HIP_TEST_CASE(Unit_hipGLGetDevices_Positive_EGLAfterNoCurrentContext) {
  CHECK_IMAGE_SUPPORT

  if (GetGLContextType() != GLContextType::Egl) {
    HIP_SKIP_TEST("EGL recovery requires GL_CONTEXT_TYPE=EGL");
  }

  // Explicitly clear both kinds of current context; this case must also work when run after
  // another GL test in the same process.
  if (glXGetCurrentContext() != nullptr) {
    REQUIRE(glXMakeCurrent(glXGetCurrentDisplay(), None, nullptr));
  }
  REQUIRE(eglBindAPI(EGL_OPENGL_API));
  const EGLDisplay previous_display = eglGetCurrentDisplay();
  if (previous_display != EGL_NO_DISPLAY) {
    REQUIRE(eglMakeCurrent(previous_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT));
  }
  REQUIRE(eglGetCurrentContext() == EGL_NO_CONTEXT);

  const int device_count = HipTest::getDeviceCount();
  REQUIRE(device_count > 0);
  unsigned int gl_device_count = 0;
  std::vector<int> gl_devices(device_count, -1);
  HIP_CHECK_ERROR(
      hipGLGetDevices(&gl_device_count, gl_devices.data(), device_count, hipGLDeviceListAll),
      hipErrorInvalidValue);

  // Failed discovery must not retain a GLX environment that poisons a later EGL association.
  EGLContextScopeGuard gl_context;
  HIP_CHECK(hipGLGetDevices(&gl_device_count, gl_devices.data(), device_count, hipGLDeviceListAll));
  REQUIRE(gl_device_count == 1);
  REQUIRE(gl_devices.at(0) == 0);
}
#endif

HIP_TEST_CASE(Unit_hipGLGetDevices_Positive_Parameters) {
  CHECK_IMAGE_SUPPORT
  
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

HIP_TEST_CASE(Unit_hipGLGetDevices_Negative_Parameters) {
  CHECK_IMAGE_SUPPORT

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
