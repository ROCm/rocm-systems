/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#if defined(__HIP_PLATFORM_AMD__) || CUDA_VERSION < CUDA_12000

texture<float, 1, hipReadModeElementType> tex;

HIP_TEST_CASE(Unit_hipTexRefSetAddress_Basic) {
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

HIP_TEST_CASE(Unit_hipTexRefSetAddress_Positive) {
  CHECK_IMAGE_SUPPORT
  hipModule_t module = nullptr;
  hipTexRef tex_ref = nullptr;
  float* tex_buffer = nullptr;
  size_t offset = 0, tex_size = sizeof(float);

  HIP_CHECK(hipFree(nullptr));
  HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
  HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));
  HIP_CHECK(hipMalloc(&tex_buffer, sizeof(float)));

  SECTION("offset is null") {
    HIP_CHECK(hipTexRefSetAddress(nullptr, tex_ref, reinterpret_cast<hipDeviceptr_t>(tex_buffer),
                                  tex_size));
  }

  SECTION("size is 0") {
    HIP_CHECK(
        hipTexRefSetAddress(&offset, tex_ref, reinterpret_cast<hipDeviceptr_t>(tex_buffer), 0));
  }

  HIP_CHECK(hipModuleUnload(module));
  HIP_CHECK(hipFree(tex_buffer));
}

HIP_TEST_CASE(Unit_hipTexRefSetAddress_Negative) {
  CHECK_IMAGE_SUPPORT
  hipModule_t module = nullptr;
  hipTexRef tex_ref = nullptr;
  float* tex_buffer = nullptr;
  size_t offset = 0, tex_size = sizeof(float);

  HIP_CHECK(hipFree(nullptr));
  HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
  HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));
  HIP_CHECK(hipMalloc(&tex_buffer, sizeof(float)));

#if HT_AMD
  HIP_CHECK_ERROR(
      hipTexRefSetAddress(&offset, nullptr, reinterpret_cast<hipDeviceptr_t>(tex_buffer), tex_size),
      hipErrorInvalidValue);
#else
  HIP_CHECK_ERROR(
      hipTexRefSetAddress(&offset, nullptr, reinterpret_cast<hipDeviceptr_t>(tex_buffer), tex_size),
      hipErrorInvalidResourceHandle);
#endif

  HIP_CHECK(hipModuleUnload(module));
  HIP_CHECK(hipFree(tex_buffer));
}

#endif
