/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "hip_module_common.hh"

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <utils.hh>

static hipModule_t GetModule() {
  HIP_CHECK(hipFree(nullptr));
  static const auto mg = ModuleGuard::LoadModule("get_function_module.code");
  return mg.module();
}

TEST_CASE("Unit_hipKernelGetAttribute_Positive_Basic") {
  hipKernel_t kernel = GetKernel(GetModule(), "GlobalKernel");

  int pi;
  int device_id = 0;

  SECTION("binaryVersion") {
    HIP_CHECK(hipKernelGetAttribute(&pi, HIP_FUNC_ATTRIBUTE_BINARY_VERSION, kernel, device_id));
    const auto major = GetDeviceAttribute(hipDeviceAttributeComputeCapabilityMajor, 0);
    const auto minor = GetDeviceAttribute(hipDeviceAttributeComputeCapabilityMinor, 0);
    REQUIRE(pi == major * 10 + minor);
  }

  SECTION("cacheModeCA") {
    HIP_CHECK(hipKernelGetAttribute(&pi, HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA, kernel, device_id));
    REQUIRE((pi == 0 || pi == 1));
  }

  SECTION("maxThreadsPerBlock") {
    HIP_CHECK(
        hipKernelGetAttribute(&pi, HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kernel, device_id));
    REQUIRE(pi == GetDeviceAttribute(hipDeviceAttributeMaxThreadsPerBlock, 0));
  }

  SECTION("numRegs") {
    HIP_CHECK(hipKernelGetAttribute(&pi, HIP_FUNC_ATTRIBUTE_NUM_REGS, kernel, device_id));
    REQUIRE(pi >= 0);
  }

  SECTION("ptxVersion") {
    HIP_CHECK(hipKernelGetAttribute(&pi, HIP_FUNC_ATTRIBUTE_PTX_VERSION, kernel, device_id));
    REQUIRE(pi > 0);
  }

  SECTION("sharedSizeBytes") {
    HIP_CHECK(hipKernelGetAttribute(&pi, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel, device_id));
    REQUIRE(pi <= GetDeviceAttribute(hipDeviceAttributeMaxSharedMemoryPerBlock, 0));
  }
}

TEST_CASE("Unit_hipKernelGetAttribute_Negative_Parameters") {
  hipKernel_t kernel = GetKernel(GetModule(), "GlobalKernel");

  int pi;
  int device_id = 0;

  SECTION("pi == nullptr") {
    HIP_CHECK_ERROR(hipKernelGetAttribute(nullptr, HIP_FUNC_ATTRIBUTE_BINARY_VERSION, kernel, device_id),
                    hipErrorInvalidValue);
  }

  SECTION("invalid attribute") {
    HIP_CHECK_ERROR(hipKernelGetAttribute(&pi, static_cast<hipFunction_attribute>(-1), kernel, device_id),
                    hipErrorInvalidValue);
  }

  SECTION("kernel == nullptr") {
    HIP_CHECK_ERROR(hipKernelGetAttribute(&pi, HIP_FUNC_ATTRIBUTE_BINARY_VERSION, nullptr, device_id),
                    hipErrorInvalidResourceHandle);
  }
}
