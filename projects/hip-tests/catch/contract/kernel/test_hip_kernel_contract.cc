/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

namespace {
constexpr int kExpectedValue = 0x1234;

__global__ void WriteValueKernel(int* output, int value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *output = value;
  }
}

int ReadDeviceInt(int* device_ptr) {
  int value = 0;
  HIP_CHECK(hipMemcpy(&value, device_ptr, sizeof(value), hipMemcpyDeviceToHost));
  return value;
}
}

// @asserts: hipLaunchKernelGGL - a kernel launched via the triple-chevron macro runs and writes its expected output to device memory
HIP_TEST_CASE(Contract_Kernel_LaunchGGL_WritesExpectedValue) {
  hip::contract::ContractCleanup cleanup;
  int* device_value = nullptr;

  HIP_CHECK(hipMalloc(&device_value, sizeof(*device_value)));
  cleanup.Add([device_value] { (void)hipFree(device_value); });
  HIP_CHECK(hipMemset(device_value, 0, sizeof(*device_value)));

  hipLaunchKernelGGL(WriteValueKernel, dim3(1), dim3(1), 0, 0, device_value, kExpectedValue);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(ReadDeviceInt(device_value) == kExpectedValue);
}

// @asserts: hipLaunchKernel - a kernel launched via the pointer-and-args entry point runs and writes its expected output to device memory
HIP_TEST_CASE(Contract_Kernel_LaunchKernel_WritesExpectedValue) {
  hip::contract::ContractCleanup cleanup;
  int* device_value = nullptr;
  void* kernel_args[] = {&device_value, const_cast<int*>(&kExpectedValue)};

  HIP_CHECK(hipMalloc(&device_value, sizeof(*device_value)));
  cleanup.Add([device_value] { (void)hipFree(device_value); });
  HIP_CHECK(hipMemset(device_value, 0, sizeof(*device_value)));

  HIP_CHECK(hipLaunchKernel(reinterpret_cast<const void*>(WriteValueKernel), dim3(1), dim3(1),
                            kernel_args, 0, nullptr));
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(ReadDeviceInt(device_value) == kExpectedValue);
}

// @asserts: hipGetLastError - a valid kernel launch leaves the last-error state clear (returns hipSuccess)
HIP_TEST_CASE(Contract_Kernel_GetLastErrorAfterValidLaunch_ReturnsSuccess) {
  hip::contract::ContractCleanup cleanup;
  int* device_value = nullptr;

  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMalloc(&device_value, sizeof(*device_value)));
  cleanup.Add([device_value] { (void)hipFree(device_value); });

  hipLaunchKernelGGL(WriteValueKernel, dim3(1), dim3(1), 0, 0, device_value, kExpectedValue);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
}

// @asserts: hipGetLastError - a launch with an invalid launch configuration records a non-success error retrievable via hipGetLastError
HIP_TEST_CASE(Contract_Kernel_InvalidConfiguration_RecordsReturnedError) {
  hip::contract::ContractCleanup cleanup;
  int* device_value = nullptr;

  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMalloc(&device_value, sizeof(*device_value)));
  cleanup.Add([device_value] { (void)hipFree(device_value); });

  hipLaunchKernelGGL(WriteValueKernel, dim3(1), dim3(0), 0, 0, device_value, kExpectedValue);
  const hipError_t error = hipGetLastError();

  REQUIRE(error != hipSuccess);
  HIP_CHECK(hipGetLastError());
}
