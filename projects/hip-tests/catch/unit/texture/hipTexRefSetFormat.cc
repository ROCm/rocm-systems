/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#if defined(__HIP_PLATFORM_AMD__) || CUDA_VERSION < CUDA_12000

texture<float, 2, hipReadModeElementType> tex;

TEST_CASE(Unit_hipTexRefSetFormat_Positive) {
  CHECK_IMAGE_SUPPORT
  hipModule_t module = nullptr;
  hipTexRef tex_ref = nullptr;
  int num_channels = 0;
  hipArray_Format format_get;

  hipArray_Format format_set = HIP_AD_FORMAT_UNSIGNED_INT32;

  HIP_CHECK(hipFree(nullptr));
  HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
  HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));

  HIP_CHECK(hipTexRefSetFormat(tex_ref, format_set, num_channels));
  HIP_CHECK(hipTexRefGetFormat(&format_get, &num_channels, tex_ref));
  REQUIRE(format_get == format_set);
  REQUIRE(num_channels == 0);

  HIP_CHECK(hipModuleUnload(module));
}

TEST_CASE(Unit_hipTexRefSetFormat_Negative) {
  CHECK_IMAGE_SUPPORT
  hipModule_t module = nullptr;
  hipTexRef tex_ref = nullptr;
  int num_channels = 1;
  hipArray_Format format = HIP_AD_FORMAT_UNSIGNED_INT32;

  HIP_CHECK(hipFree(nullptr));
  HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
  HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));

#if HT_AMD
  HIP_CHECK_ERROR(hipTexRefSetFormat(nullptr, format, num_channels), hipErrorInvalidValue);
#else
  HIP_CHECK_ERROR(hipTexRefSetFormat(nullptr, format, num_channels), hipErrorInvalidResourceHandle);
#endif

  HIP_CHECK(hipModuleUnload(module));
}

#endif  // __HIP_PLATFORM_AMD__ || CUDA_VERSION < CUDA_12000
