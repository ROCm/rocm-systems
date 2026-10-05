/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#if !HT_NVIDIA

namespace {
void LoadLibraryKernel(hipLibrary_t* library, hipKernel_t* kernel) {
  std::string code_object = "library_code_load.code";
  HIP_CHECK(hipLibraryLoadFromFile(library, code_object.c_str(), nullptr, nullptr, 0, nullptr,
                                   nullptr, 0));
  HIP_CHECK(hipLibraryGetKernel(kernel, *library, "add_kernel"));
}
}  // namespace

HIP_TEST_CASE(Unit_hipKernelSetCacheConfig_Positive) {
  HIP_CHECK(hipSetDevice(0));
  hipLibrary_t library = nullptr;
  hipKernel_t kernel = nullptr;
  LoadLibraryKernel(&library, &kernel);
  hipFuncCache_t carveouts[] = { hipFuncCachePreferNone,
                                  hipFuncCachePreferShared,
                                  hipFuncCachePreferL1,
                                  hipFuncCachePreferEqual };
  hipDevice_t device;
  int currentDevice = -1;

  HIP_CHECK(hipGetDevice(&currentDevice));
  HIP_CHECK(hipDeviceGet(&device, currentDevice));

  for (const auto& carveout : carveouts) {
    HIP_CHECK(hipKernelSetCacheConfig(kernel, carveout, device));
  }
}
#endif
// TODO g-h-c implement contract test
