/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include "hip_module_common.hh"

#include <hip/hip_runtime_api.h>
#include <resource_guards.hh>
#include <utils.hh>

using ExtModuleLaunchKernelSig = hipError_t(hipFunction_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                            uint32_t, uint32_t, size_t, hipStream_t, void**, void**,
                                            hipEvent_t, hipEvent_t, uint32_t);

template <ExtModuleLaunchKernelSig* func> void ModuleLaunchKernelPositiveBasic() {
  auto mg = ModuleGuard::InitModule("launch_kernel_module.code");
  SECTION("Kernel with no arguments") {
    hipFunction_t f = GetKernel(mg.module(), "NOPKernel");
    HIP_CHECK(func(f, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0u));
    HIP_CHECK(hipDeviceSynchronize());
  }

  SECTION("Kernel with arguments using kernelParams") {
    hipFunction_t f = GetKernel(mg.module(), "Kernel42");
    LinearAllocGuard<int> result_dev(LinearAllocs::hipMalloc, sizeof(int));
    HIP_CHECK(hipMemset(result_dev.ptr(), 0, sizeof(*result_dev.ptr())));
    int* result_ptr = result_dev.ptr();
    void* kernel_args[1] = {&result_ptr};
    HIP_CHECK(func(f, 1, 1, 1, 1, 1, 1, 0, nullptr, kernel_args, nullptr, nullptr, nullptr, 0u));
    int result = 0;
    HIP_CHECK(hipMemcpy(&result, result_dev.ptr(), sizeof(result), hipMemcpyDefault));
    REQUIRE(result == 42);
  }

  SECTION("Kernel with arguments using extra") {
    hipFunction_t f = GetKernel(mg.module(), "Kernel42");
    LinearAllocGuard<int> result_dev(LinearAllocs::hipMalloc, sizeof(int));
    HIP_CHECK(hipMemset(result_dev.ptr(), 0, sizeof(*result_dev.ptr())));
    int* result_ptr = result_dev.ptr();
    size_t size = sizeof(result_ptr);
    // clang-format off
    void *extra[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &result_ptr,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
        HIP_LAUNCH_PARAM_END
    };
    // clang-format on
    HIP_CHECK(func(f, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, extra, nullptr, nullptr, 0u));
    int result = 0;
    HIP_CHECK(hipMemcpy(&result, result_dev.ptr(), sizeof(result), hipMemcpyDefault));
    REQUIRE(result == 42);
  }
}

template <ExtModuleLaunchKernelSig* func> void ModuleLaunchKernelPositiveParameters() {
  const auto LaunchNOPKernel = [=](unsigned int gridDimX, unsigned int gridDimY,
                                   unsigned int gridDimZ, unsigned int blockDimX,
                                   unsigned int blockDimY, unsigned int blockDimZ) {
    auto mg = ModuleGuard::InitModule("launch_kernel_module.code");
    hipFunction_t f = GetKernel(mg.module(), "NOPKernel");
    HIP_CHECK(func(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, nullptr,
                   nullptr, nullptr, nullptr, nullptr, 0u));
    HIP_CHECK(hipDeviceSynchronize());
  };

  SECTION("gridDimX == maxGridDimX") {
    const unsigned int x = GetDeviceAttribute(hipDeviceAttributeMaxGridDimX, 0);
    LaunchNOPKernel(x, 1, 1, 1, 1, 1);
  }

  SECTION("gridDimY == maxGridDimY") {
    const unsigned int y = GetDeviceAttribute(hipDeviceAttributeMaxGridDimY, 0);
    LaunchNOPKernel(1, y, 1, 1, 1, 1);
  }

  SECTION("gridDimZ == maxGridDimZ") {
    const unsigned int z = GetDeviceAttribute(hipDeviceAttributeMaxGridDimZ, 0);
    LaunchNOPKernel(1, 1, z, 1, 1, 1);
  }

  SECTION("blockDimX == maxBlockDimX") {
    const unsigned int x = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimX, 0);
    LaunchNOPKernel(1, 1, 1, x, 1, 1);
  }

  SECTION("blockDimY == maxBlockDimY") {
    const unsigned int y = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimY, 0);
    LaunchNOPKernel(1, 1, 1, 1, y, 1);
  }

  SECTION("blockDimZ == maxBlockDimZ") {
    const unsigned int z = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimZ, 0);
    LaunchNOPKernel(1, 1, 1, 1, 1, z);
  }
}

template <ExtModuleLaunchKernelSig* func> void ModuleLaunchKernelNegativeParameters(
                                                           bool extLaunch = false) {
  auto mg = ModuleGuard::InitModule("launch_kernel_module.code");
  hipFunction_t f = GetKernel(mg.module(), "NOPKernel");
  hipError_t expectedErrorLaunchParam = (extLaunch == true) ? hipErrorInvalidConfiguration
                                                             : hipErrorInvalidValue;
  hipError_t expectedErrorOverCapacityGridDim = (extLaunch == true) ? hipSuccess
                                                                    : hipErrorInvalidValue;

  SECTION("f == nullptr") {
    HIP_CHECK_ERROR(
        func(nullptr, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0u),
        hipErrorInvalidResourceHandle);
  }

  SECTION("gridDimX == 0") {
    HIP_CHECK_ERROR(func(f, 0, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0u),
                    expectedErrorLaunchParam);
  }

  SECTION("gridDimY == 0") {
    HIP_CHECK_ERROR(func(f, 1, 0, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0u),
                    expectedErrorLaunchParam);
  }

  SECTION("gridDimZ == 0") {
    HIP_CHECK_ERROR(func(f, 1, 1, 0, 1, 1, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0u),
                    expectedErrorLaunchParam);
  }

  SECTION("blockDimX == 0") {
    HIP_CHECK_ERROR(func(f, 1, 1, 1, 0, 1, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0u),
                    expectedErrorLaunchParam);
  }

  SECTION("blockDimY == 0") {
    HIP_CHECK_ERROR(func(f, 1, 1, 1, 1, 0, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0u),
                    expectedErrorLaunchParam);
  }

  SECTION("blockDimZ == 0") {
    HIP_CHECK_ERROR(func(f, 1, 1, 1, 1, 1, 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0u),
                    expectedErrorLaunchParam);
  }

  SECTION("gridDimX > maxGridDimX") {
    const unsigned int x = GetDeviceAttribute(hipDeviceAttributeMaxGridDimX, 0) + 1u;
    HIP_CHECK_ERROR(func(f, x, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0u),
                    expectedErrorOverCapacityGridDim);
  }

  SECTION("gridDimY > maxGridDimY") {
    const unsigned int y = GetDeviceAttribute(hipDeviceAttributeMaxGridDimY, 0) + 1u;
    HIP_CHECK_ERROR(func(f, 1, y, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0u),
                    expectedErrorOverCapacityGridDim);
  }

  SECTION("gridDimZ > maxGridDimZ") {
    const unsigned int z = GetDeviceAttribute(hipDeviceAttributeMaxGridDimZ, 0) + 1u;
    HIP_CHECK_ERROR(func(f, 1, 1, z, 1, 1, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0u),
                    expectedErrorOverCapacityGridDim);
  }

  SECTION("blockDimX > maxBlockDimX") {
    const unsigned int x = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimX, 0) + 1u;
    HIP_CHECK_ERROR(func(f, 1, 1, 1, x, 1, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0u),
                    expectedErrorLaunchParam);
  }

  SECTION("blockDimY > maxBlockDimY") {
    const unsigned int y = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimY, 0) + 1u;
    HIP_CHECK_ERROR(func(f, 1, 1, 1, 1, y, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0u),
                    expectedErrorLaunchParam);
  }

  SECTION("blockDimZ > maxBlockDimZ") {
    const unsigned int z = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimZ, 0) + 1u;
    HIP_CHECK_ERROR(func(f, 1, 1, 1, 1, 1, z, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0u),
                    expectedErrorLaunchParam);
  }

  SECTION("blockDimX * blockDimY * blockDimZ > MaxThreadsPerBlock") {
    const unsigned int max = GetDeviceAttribute(hipDeviceAttributeMaxThreadsPerBlock, 0);
    const unsigned int dim = std::ceil(std::cbrt(max)) + 1;
    HIP_CHECK_ERROR(
        func(f, 1, 1, 1, dim, dim, dim, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0u),
        expectedErrorLaunchParam);
  }

  SECTION("sharedMemBytes > max shared memory per block") {
    const unsigned int max = GetDeviceAttribute(hipDeviceAttributeMaxSharedMemoryPerBlock, 0) + 1u;
    HIP_CHECK_ERROR(func(f, 1, 1, 1, 1, 1, 1, max, nullptr, nullptr, nullptr, nullptr, nullptr, 0u),
                    hipErrorInvalidValue);
  }

  SECTION("Passing kernel_args and extra simultaneously") {
    auto mg = ModuleGuard::InitModule("launch_kernel_module.code");
    hipFunction_t f = GetKernel(mg.module(), "Kernel42");
    LinearAllocGuard<int> result_dev(LinearAllocs::hipMalloc, sizeof(int));
    int* result_ptr = result_dev.ptr();
    size_t size = sizeof(result_ptr);
    void* kernel_args[1] = {&result_ptr};
    // clang-format off
    void *extra[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &result_ptr,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
        HIP_LAUNCH_PARAM_END
    };
    // clang-format on
    HIP_CHECK_ERROR(func(f, 1, 1, 1, 1, 1, 1, 0, nullptr, kernel_args, extra, nullptr, nullptr, 0u),
                    hipErrorInvalidValue);
  }

  SECTION("Stream not on the same device") {
    int numDevices = 0;
    HIP_CHECK(hipGetDeviceCount(&numDevices));
    if (numDevices < 2) {
      SUCCEED("skipped the testcase as number of devices is less than 2");
    } else {
      HIP_CHECK(hipSetDevice(1));
      hipStream_t s1;
      HIP_CHECK(hipStreamCreate(&s1));
      HIP_CHECK(hipSetDevice(0));
      hipFunction_t f = GetKernel(mg.module(), "Kernel42");
      void* extra[0] = {};
      HIP_CHECK_ERROR(func(f, 1, 1, 1, 1, 1, 1, 0, s1, nullptr, extra, nullptr, nullptr, 0u),
                      hipErrorInvalidResourceHandle);
      HIP_CHECK(hipStreamDestroy(s1));
    }
  }

  SECTION("Invalid extra") {
    auto mg = ModuleGuard::InitModule("launch_kernel_module.code");
    hipFunction_t f = GetKernel(mg.module(), "Kernel42");
    void* extra[0] = {};
    HIP_CHECK_ERROR(func(f, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, extra, nullptr, nullptr, 0u),
                    hipErrorInvalidValue);
  }
}
