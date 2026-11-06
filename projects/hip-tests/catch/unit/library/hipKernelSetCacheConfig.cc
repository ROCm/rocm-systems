/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#if !HT_NVIDIA || CUDA_VERSION >= CUDA_12000

namespace {
void LoadLibraryKernel(hipLibrary_t* library, hipKernel_t* kernel) {
  std::string code_object = "library_code_load.code";
  HIP_CHECK(hipLibraryLoadFromFile(library, code_object.c_str(), nullptr, nullptr, 0, nullptr,
                                   nullptr, 0));
  HIP_CHECK(hipLibraryGetKernel(kernel, *library, "add_kernel"));
}
}  // namespace

HIP_TEST_CASE(Unit_hipKernelSetCacheConfig_Positive) {
  hipLibrary_t library = nullptr;
  hipKernel_t kernel = nullptr;
  hipFuncCache_t carveouts[] = { hipFuncCachePreferNone,
                                 hipFuncCachePreferShared,
                                 hipFuncCachePreferL1,
                                 hipFuncCachePreferEqual };
  hipDevice_t device;
  int currentDevice = -1;

  LoadLibraryKernel(&library, &kernel);
  HIP_CHECK(hipGetDevice(&currentDevice));
  HIP_CHECK(hipDeviceGet(&device, currentDevice));

  for (const auto& carveout : carveouts) {
    HIP_CHECK(hipKernelSetCacheConfig(kernel, carveout, device));
  }
}

HIP_TEST_CASE(Unit_hipKernelSetCacheConfig_Negative) {
  hipKernel_t kernel = nullptr;
  int currentDevice = -1;
  hipDevice_t device;
  auto unloadLibrary = [](hipLibrary_t* ptr) {
          if (ptr) {
            HIP_CHECK(hipLibraryUnload(*ptr));
          };
        };
  std::unique_ptr<hipLibrary_t, decltype(unloadLibrary)> libraryPtr(nullptr, unloadLibrary);
  hipLibrary_t library;

  LoadLibraryKernel(&library, &kernel);
  libraryPtr.reset(&library);
  HIP_CHECK(hipGetDevice(&currentDevice));
  HIP_CHECK(hipDeviceGet(&device, currentDevice));

  SECTION("invalid ordinal") {
    HIP_CHECK_ERROR(hipKernelSetCacheConfig(kernel, static_cast<hipFuncCache_t>(4), device),
                  hipErrorInvalidValue);
  }

  SECTION("invalid library") {
    HIP_CHECK(hipLibraryUnload(library));
    libraryPtr.release();
    HIP_CHECK_ERROR(hipKernelSetCacheConfig(kernel, hipFuncCachePreferEqual, device),
                  hipErrorInvalidResourceHandle);
  }

  SECTION("invalid kernel") {
    hipKernel_t kernel2 = nullptr;

    HIP_CHECK_ERROR(hipKernelSetCacheConfig(kernel2, hipFuncCachePreferEqual, device),
                  hipErrorInvalidResourceHandle);
  }

  SECTION("invalid device") {
    hipDevice_t device2 = -1;

    HIP_CHECK_ERROR(hipKernelSetCacheConfig(kernel, hipFuncCachePreferEqual, device2),
                  hipErrorInvalidDevice);
  }
}
#endif
