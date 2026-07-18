/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

// BACKEND-DIFF: The legacy call-configuration launch path (hipConfigureCall,
// hipSetupArgument, hipLaunchByPtr) is AMD-only; the NVIDIA backend does not
// expose these entry points, so this whole translation unit builds only on AMD.
// Parity would require NVIDIA to provide these legacy launch entry points (or
// the tests to be re-expressed on the portable launch APIs).
#if HT_AMD

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
}  // namespace

HIP_TEST_CASE(Contract_CallConfig_ConfigureSetupLaunch_WritesExpectedValue) {
  hip::contract::ContractCleanup cleanup;
  int* device_value = nullptr;
  HIP_CHECK(hipMalloc(&device_value, sizeof(*device_value)));
  cleanup.Add([&] { (void)hipFree(device_value); });
  HIP_CHECK(hipMemset(device_value, 0, sizeof(*device_value)));

  // The legacy call-configuration path stages a launch in three steps: configure
  // the grid/block, push each argument onto the argument stack at its natural
  // offset, then launch the kernel by its host-side function pointer. A
  // single-thread grid keeps the write deterministic.
  HIP_CHECK(hipConfigureCall(dim3(1), dim3(1), 0, nullptr));

  int value = kExpectedValue;
  HIP_CHECK(hipSetupArgument(&device_value, sizeof(device_value), 0));
  HIP_CHECK(hipSetupArgument(&value, sizeof(value), sizeof(device_value)));

  HIP_CHECK(hipLaunchByPtr(reinterpret_cast<const void*>(WriteValueKernel)));
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(ReadDeviceInt(device_value) == kExpectedValue);
}

HIP_TEST_CASE(Contract_CallConfig_ConfigureSetupLaunch_RepeatedStagingIsIndependent) {
  hip::contract::ContractCleanup cleanup;
  int* first = nullptr;
  int* second = nullptr;
  HIP_CHECK(hipMalloc(&first, sizeof(*first)));
  cleanup.Add([&] { (void)hipFree(first); });
  HIP_CHECK(hipMalloc(&second, sizeof(*second)));
  cleanup.Add([&] { (void)hipFree(second); });
  HIP_CHECK(hipMemset(first, 0, sizeof(*first)));
  HIP_CHECK(hipMemset(second, 0, sizeof(*second)));

  // Each configure/setup/launch sequence stages its own argument stack. Two
  // independent launches to distinct destinations must each observe only their
  // own argument values, proving the argument stack does not leak between calls.
  const int first_value = kExpectedValue;
  HIP_CHECK(hipConfigureCall(dim3(1), dim3(1), 0, nullptr));
  HIP_CHECK(hipSetupArgument(&first, sizeof(first), 0));
  HIP_CHECK(hipSetupArgument(&first_value, sizeof(first_value), sizeof(first)));
  HIP_CHECK(hipLaunchByPtr(reinterpret_cast<const void*>(WriteValueKernel)));

  const int second_value = kExpectedValue + 1;
  HIP_CHECK(hipConfigureCall(dim3(1), dim3(1), 0, nullptr));
  HIP_CHECK(hipSetupArgument(&second, sizeof(second), 0));
  HIP_CHECK(hipSetupArgument(&second_value, sizeof(second_value), sizeof(second)));
  HIP_CHECK(hipLaunchByPtr(reinterpret_cast<const void*>(WriteValueKernel)));

  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(ReadDeviceInt(first) == first_value);
  REQUIRE(ReadDeviceInt(second) == second_value);
}
#endif  // HT_AMD
