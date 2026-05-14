/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "execution_control_common.hh"

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <resource_guards.hh>
#include <utils.hh>

HIP_TEST_CASE(Unit_hipExtLaunchMultiKernelMultiDevice_Positive_Basic) {
  const auto device_count = HipTest::getDeviceCount();

  std::vector<hipLaunchParams> params_list(device_count);

  int device = 0;
  for (auto& params : params_list) {
    params.func = reinterpret_cast<void*>(kernel);
    params.gridDim = dim3{1, 1, 1};
    params.blockDim = dim3{1, 1, 1};
    params.args = nullptr;
    params.sharedMem = 0;
    HIP_CHECK(hipSetDevice(device++));
    HIP_CHECK(hipStreamCreate(&params.stream));
  }

  HIP_CHECK(hipExtLaunchMultiKernelMultiDevice(params_list.data(), device_count, 0u));

  for (const auto params : params_list) {
    HIP_CHECK(hipStreamSynchronize(params.stream));
  }

  for (const auto params : params_list) {
    HIP_CHECK(hipStreamDestroy(params.stream));
  }
}

HIP_TEST_CASE(Unit_hipExtLaunchMultiKernelMultiDevice_Negative_Parameters) {
  const auto device_count = HipTest::getDeviceCount();

  std::vector<hipLaunchParams> params_list(device_count);

  int device = 0;
  for (auto& params : params_list) {
    params.func = reinterpret_cast<void*>(kernel);
    params.gridDim = dim3{1, 1, 1};
    params.blockDim = dim3{1, 1, 1};
    params.args = nullptr;
    params.sharedMem = 0;
    HIP_CHECK(hipSetDevice(device++));
    HIP_CHECK(hipStreamCreate(&params.stream));
  }

  SECTION("launchParamsList == nullptr") {
    HIP_CHECK_ERROR(hipExtLaunchMultiKernelMultiDevice(nullptr, device_count, 0u),
                    hipErrorInvalidValue);
  }

  SECTION("numDevices == 0") {
    HIP_CHECK_ERROR(hipExtLaunchMultiKernelMultiDevice(params_list.data(), 0, 0u),
                    hipErrorInvalidValue);
  }

  SECTION("numDevices > device count") {
    HIP_CHECK_ERROR(hipExtLaunchMultiKernelMultiDevice(params_list.data(), device_count + 1, 0u),
                    hipErrorInvalidValue);
  }

  SECTION("invalid flags") {
    HIP_CHECK_ERROR(hipExtLaunchMultiKernelMultiDevice(params_list.data(), device_count, 999),
                    hipErrorInvalidValue);
  }

  if (device_count > 1) {
    SECTION("launchParamsList.func doesn't match across all devices") {
      params_list[1].func = reinterpret_cast<void*>(kernel2);
      HIP_CHECK_ERROR(hipExtLaunchMultiKernelMultiDevice(params_list.data(), device_count, 0u),
                      hipErrorInvalidValue);
    }

    SECTION("launchParamsList.gridDim doesn't match across all kernels") {
      params_list[1].gridDim = dim3{2, 2, 2};
      HIP_CHECK_ERROR(hipExtLaunchMultiKernelMultiDevice(params_list.data(), device_count, 0u),
                      hipErrorInvalidValue);
    }

    SECTION("launchParamsList.blockDim doesn't match across all kernels") {
      params_list[1].blockDim = dim3{2, 2, 2};
      HIP_CHECK_ERROR(hipExtLaunchMultiKernelMultiDevice(params_list.data(), device_count, 0u),
                      hipErrorInvalidValue);
    }

    SECTION("launchParamsList.sharedMem doesn't match across all kernels") {
      params_list[1].sharedMem = 1024;
      HIP_CHECK_ERROR(hipExtLaunchMultiKernelMultiDevice(params_list.data(), device_count, 0u),
                      hipErrorInvalidValue);
    }
  }

  for (const auto params : params_list) {
    HIP_CHECK(hipStreamDestroy(params.stream));
  }
}

HIP_TEST_CASE(Unit_hipExtLaunchMultiKernelMultiDevice_Negative_MultiKernelSameDevice) {
  HIP_CHECK(hipSetDevice(0));

  std::vector<hipLaunchParams> params_list(2);

  for (auto& params : params_list) {
    params.func = reinterpret_cast<void*>(kernel);
    params.gridDim = dim3{1, 1, 1};
    params.blockDim = dim3{1, 1, 1};
    params.args = nullptr;
    params.sharedMem = 0;
    HIP_CHECK(hipStreamCreate(&params.stream));
  }

  HIP_CHECK_ERROR(hipExtLaunchMultiKernelMultiDevice(params_list.data(), 2, 0u),
                  hipErrorInvalidDevice);

  for (const auto params : params_list) {
    HIP_CHECK(hipStreamDestroy(params.stream));
  }
}
