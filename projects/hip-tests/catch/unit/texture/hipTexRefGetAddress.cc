/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#if defined(__HIP_PLATFORM_AMD__) || CUDA_VERSION < CUDA_12000

texture<float, 1, hipReadModeElementType> tex;

HIP_TEST_CASE(Unit_hipTexRefGetAddress_Positive) {
  CHECK_IMAGE_SUPPORT
  hipDeviceptr_t device_ptr;
  hipModule_t module = nullptr;
  hipTexRef tex_ref = nullptr;
  float* tex_buffer = nullptr;
  size_t offset = 0, tex_size = sizeof(float);

  HIP_CHECK(hipFree(nullptr));
  HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
  HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));
  HIP_CHECK(hipMalloc(&tex_buffer, sizeof(float)));
  HIP_CHECK(hipTexRefSetAddress(&offset, tex_ref, reinterpret_cast<hipDeviceptr_t>(tex_buffer),
                                tex_size));
  HIP_CHECK(hipTexRefGetAddress(&device_ptr, tex_ref));

  REQUIRE(reinterpret_cast<void*>(device_ptr) != nullptr);
  HIP_CHECK(hipModuleUnload(module));
  HIP_CHECK(hipFree(tex_buffer));
}

HIP_TEST_CASE(Unit_hipTexRefGetAddress_Negative) {
  CHECK_IMAGE_SUPPORT
  hipDeviceptr_t device_ptr;
  hipModule_t module = nullptr;
  hipTexRef tex_ref = nullptr;
  float* tex_buffer = nullptr;
  size_t offset = 0, tex_size = sizeof(float);

  HIP_CHECK(hipFree(nullptr));
  HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
  HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));
  HIP_CHECK(hipMalloc(&tex_buffer, sizeof(float)));
  HIP_CHECK(hipTexRefSetAddress(&offset, tex_ref, reinterpret_cast<hipDeviceptr_t>(tex_buffer),
                                tex_size));

  SECTION("texture reference is null") {
#if HT_AMD
    HIP_CHECK_ERROR(hipTexRefGetAddress(&device_ptr, nullptr), hipErrorInvalidValue);
#else
    HIP_CHECK_ERROR(hipTexRefGetAddress(&device_ptr, nullptr), hipErrorInvalidResourceHandle);
#endif
  }

  HIP_CHECK(hipModuleUnload(module));
  HIP_CHECK(hipFree(tex_buffer));
}

HIP_TEST_CASE(Unit_hipTexRefGetAddress_AdressNotSet) {
  CHECK_IMAGE_SUPPORT
  hipDeviceptr_t device_ptr;
  hipModule_t module = nullptr;
  hipTexRef tex_ref = nullptr;
  float* tex_buffer = nullptr;

  HIP_CHECK(hipFree(nullptr));
  HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
  HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));
  HIP_CHECK(hipMalloc(&tex_buffer, sizeof(float)));
  HIP_CHECK_ERROR(hipTexRefGetAddress(&device_ptr, tex_ref), hipErrorInvalidValue);

  HIP_CHECK(hipModuleUnload(module));
  HIP_CHECK(hipFree(tex_buffer));
}

#endif
