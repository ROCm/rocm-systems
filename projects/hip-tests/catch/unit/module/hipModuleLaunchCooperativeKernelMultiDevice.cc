/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

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

/**
 * @addtogroup hipModuleLaunchCooperativeKernelMultiDevice
 * hipModuleLaunchCooperativeKernelMultiDevice
 * @{
 * @ingroup ModuleTest
 * `hipModuleLaunchCooperativeKernelMultiDevice(hipFunctionLaunchParams* launchParamsList, unsigned
 * int numDevices, unsigned int flags)` -
 * Launches kernels on multiple devices where thread blocks can cooperate and synchronize as they
 * execute.
 */

#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <utils.hh>

#include "hip_module_launch_kernel_common.hh"

/**
 * Test Description
 * ------------------------
 *  - Tests `hipModuleLaunchCooperativeKernel` for a cooperative kernel with no parameters.
 * Test source
 * ------------------------
 *  - unit/module/hipModuleLaunchCooperativeKernelMultiDevice.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.5
 */
TEST_CASE("Unit_hipModuleLaunchCooperativeKernelMultiDevice_Positive_Basic") {
  auto mg = ModuleGuard::InitModule("launch_kernel_module.code");
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }

  const auto device_count = HipTest::getDeviceCount();

  std::vector<hipFunctionLaunchParams> params_list(device_count);
  std::vector<hipStream_t> streams_list(device_count);
  std::vector<hipModule_t> modules_list(device_count);

  for (int index = 0; index < device_count; index++) {
    HIP_CHECK(hipSetDevice(index));
    HIP_CHECK(hipStreamCreate(&streams_list[index]));
    HIP_CHECK(hipModuleLoad(&modules_list[index], "launch_kernel_module.code"));
  }

  HIP_CHECK(hipSetDevice(0));
  for (int index = 0; index < device_count; index++) {
    hipFunction_t kernelFunction;
    HIP_CHECK(hipModuleGetFunction(&kernelFunction, modules_list[index], "CoopKernel"));
    params_list[index].function = kernelFunction;
    params_list[index].gridDimX = 1;
    params_list[index].gridDimY = 1;
    params_list[index].gridDimZ = 1;
    params_list[index].blockDimX = 1;
    params_list[index].blockDimY = 1;
    params_list[index].blockDimZ = 1;
    params_list[index].kernelParams = nullptr;
    params_list[index].sharedMemBytes = 0;
    params_list[index].hStream = streams_list[index];
  }

  HIP_CHECK(hipModuleLaunchCooperativeKernelMultiDevice(params_list.data(), device_count, 0u));

  for (const auto params : params_list) {
    HIP_CHECK(hipStreamSynchronize(params.hStream));
  }

  for (int index = 0; index < device_count; index++) {
    HIP_CHECK(hipStreamDestroy(params_list[index].hStream));
    HIP_CHECK(hipModuleUnload(modules_list[index]));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Negative parameters test for `hipModuleLaunchCooperativeKernelMultiDevice`.
 * Test source
 * ------------------------
 *  - unit/module/hipModuleLaunchCooperativeKernelMultiDevice.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.5
 */
TEST_CASE("Unit_hipModuleLaunchCooperativeKernelMultiDevice_Negative_Parameters") {
  auto mg = ModuleGuard::InitModule("launch_kernel_module.code");
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }

  const auto device_count = HipTest::getDeviceCount();

  std::vector<hipFunctionLaunchParams> params_list(device_count);
  std::vector<hipStream_t> streams_list(device_count);
  std::vector<hipModule_t> modules_list(device_count);

  for (int index = 0; index < device_count; index++) {
    HIP_CHECK(hipSetDevice(index));
    HIP_CHECK(hipStreamCreate(&streams_list[index]));
    HIP_CHECK(hipModuleLoad(&modules_list[index], "launch_kernel_module.code"));
  }

  HIP_CHECK(hipSetDevice(0));
  for (int index = 0; index < device_count; index++) {
    hipFunction_t kernelFunction;
    HIP_CHECK(hipModuleGetFunction(&kernelFunction, modules_list[index], "CoopKernel"));
    params_list[index].function = kernelFunction;
    params_list[index].gridDimX = 1;
    params_list[index].gridDimY = 1;
    params_list[index].gridDimZ = 1;
    params_list[index].blockDimX = 1;
    params_list[index].blockDimY = 1;
    params_list[index].blockDimZ = 1;
    params_list[index].kernelParams = nullptr;
    params_list[index].sharedMemBytes = 0;
    params_list[index].hStream = streams_list[index];
  }

  SECTION("launchParamsList == nullptr") {
    HIP_CHECK_ERROR(hipModuleLaunchCooperativeKernelMultiDevice(nullptr, device_count, 0u),
                    hipErrorInvalidValue);
  }

  SECTION("numDevices == 0") {
    HIP_CHECK_ERROR(hipModuleLaunchCooperativeKernelMultiDevice(params_list.data(), 0, 0u),
                    hipErrorInvalidValue);
  }

  SECTION("numDevices > device count") {
    HIP_CHECK_ERROR(
        hipModuleLaunchCooperativeKernelMultiDevice(params_list.data(), device_count + 1, 0u),
        hipErrorInvalidValue);
  }

  SECTION("invalid flags") {
    HIP_CHECK_ERROR(
        hipModuleLaunchCooperativeKernelMultiDevice(params_list.data(), device_count, 999),
        hipErrorInvalidValue);
  }

  if (device_count > 1) {
    SECTION("launchParamsList.func doesn't match across all devices") {
      params_list[1].function = GetKernel(mg.module(), "NOPKernel");
#if HT_AMD
      HIP_CHECK_ERROR(
          hipModuleLaunchCooperativeKernelMultiDevice(params_list.data(), device_count, 0u),
          hipErrorInvalidDevice);
#else
      HIP_CHECK_ERROR(
          hipModuleLaunchCooperativeKernelMultiDevice(params_list.data(), device_count, 0u),
          hipErrorInvalidResourceHandle);
#endif
    }

    SECTION("launchParamsList.gridDim doesn't match across all kernels") {
      params_list[1].gridDimX = 2;
      HIP_CHECK_ERROR(
          hipModuleLaunchCooperativeKernelMultiDevice(params_list.data(), device_count, 0u),
          hipErrorInvalidValue);
    }

    SECTION("launchParamsList.blockDim doesn't match across all kernels") {
      params_list[1].blockDimX = 2;
      HIP_CHECK_ERROR(
          hipModuleLaunchCooperativeKernelMultiDevice(params_list.data(), device_count, 0u),
          hipErrorInvalidValue);
    }

    SECTION("launchParamsList.sharedMem doesn't match across all kernels") {
      params_list[1].sharedMemBytes = 1024;
      HIP_CHECK_ERROR(
          hipModuleLaunchCooperativeKernelMultiDevice(params_list.data(), device_count, 0u),
          hipErrorInvalidValue);
    }
    SECTION("Stream doesn't match across all devices") {
      HIP_CHECK(hipStreamDestroy(params_list[1].hStream));
      HIP_CHECK(hipStreamCreate(&params_list[1].hStream));
#if HT_AMD
      HIP_CHECK_ERROR(
          hipModuleLaunchCooperativeKernelMultiDevice(params_list.data(), device_count, 0u),
          hipErrorInvalidDevice);
#else
      HIP_CHECK_ERROR(
          hipModuleLaunchCooperativeKernelMultiDevice(params_list.data(), device_count, 0u),
          hipErrorInvalidResourceHandle);
#endif
    }
  }

  for (int index = 0; index < device_count; index++) {
    HIP_CHECK(hipStreamDestroy(params_list[index].hStream));
    HIP_CHECK(hipModuleUnload(modules_list[index]));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Tries running `hipModuleLaunchCooperativeKernelMultiDevice` with multiple kernels on the same
 * device.
 * Test source
 * ------------------------
 *  - unit/module/hipModuleLaunchCooperativeKernelMultiDevice.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.5
 */
TEST_CASE("Unit_hipModuleLaunchCooperativeKernelMultiDevice_Negative_MultiKernelSameDevice") {
  auto mg = ModuleGuard::InitModule("launch_kernel_module.code");
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }

  int device_count;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count < 2) {
    HipTest::HIP_SKIP_TEST("Test requires more than one device");
    return;
  }
  std::vector<hipFunctionLaunchParams> params_list(device_count);
  std::vector<hipModule_t> modules_list(device_count);

  for (int index = 0; index < device_count; index++) {
    HIP_CHECK(hipSetDevice(index));
    HIP_CHECK(hipModuleLoad(&modules_list[index], "launch_kernel_module.code"));
  }

  HIP_CHECK(hipSetDevice(0));
  for (int index = 0; index < device_count; index++) {
    hipFunction_t kernelFunction;
    HIP_CHECK(hipModuleGetFunction(&kernelFunction, modules_list[index], "CoopKernel"));
    params_list[index].function = kernelFunction;
    params_list[index].gridDimX = 1;
    params_list[index].gridDimY = 1;
    params_list[index].gridDimZ = 1;
    params_list[index].blockDimX = 1;
    params_list[index].blockDimY = 1;
    params_list[index].blockDimZ = 1;
    params_list[index].kernelParams = nullptr;
    params_list[index].sharedMemBytes = 0;
    HIP_CHECK(hipStreamCreate(&params_list[index].hStream));
  }

#if HT_AMD
  HIP_CHECK_ERROR(hipModuleLaunchCooperativeKernelMultiDevice(params_list.data(), device_count, 0u),
                  hipErrorInvalidDevice);
#else
  HIP_CHECK_ERROR(hipModuleLaunchCooperativeKernelMultiDevice(params_list.data(), device_count, 0u),
                  hipErrorInvalidResourceHandle);
#endif

  for (int index = 0; index < device_count; index++) {
    HIP_CHECK(hipStreamDestroy(params_list[index].hStream));
    HIP_CHECK(hipModuleUnload(modules_list[index]));
  }
}

/**
 * End doxygen group ModuleTest.
 * @}
 */
