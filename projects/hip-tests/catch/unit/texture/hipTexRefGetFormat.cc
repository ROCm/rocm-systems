/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#if defined(__HIP_PLATFORM_AMD__) || CUDA_VERSION < CUDA_12000

texture<float, 2, hipReadModeElementType> tex;

TEST_CASE(Unit_hipTexRefGetFormat_Basic) {
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

TEST_CASE(Unit_hipTexRefGetFormat_Positive) {
  CHECK_IMAGE_SUPPORT
  hipModule_t module = nullptr;
  hipTexRef tex_ref = nullptr;
  int num_channels = 1;
  hipArray_Format format_get;

  hipArray_Format format_set = HIP_AD_FORMAT_UNSIGNED_INT32;

  HIP_CHECK(hipFree(nullptr));
  HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
  HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));

  HIP_CHECK(hipTexRefSetFormat(tex_ref, format_set, num_channels));

  // If format or number of channels is NULL, it will be ignored as per CUDA docs.
  SECTION("If format or numChannels is NULL, it will be ignored") {
#if HT_AMD
    HIP_CHECK_ERROR(hipTexRefGetFormat(nullptr, &num_channels, tex_ref), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipTexRefGetFormat(&format_get, nullptr, tex_ref), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipTexRefGetFormat(nullptr, nullptr, tex_ref), hipErrorInvalidValue);
#else
    HIP_CHECK(hipTexRefGetFormat(nullptr, &num_channels, tex_ref));
    HIP_CHECK(hipTexRefGetFormat(&format_get, nullptr, tex_ref));
    HIP_CHECK(hipTexRefGetFormat(nullptr, nullptr, tex_ref));
#endif
  }

  HIP_CHECK(hipModuleUnload(module));
}

TEST_CASE(Unit_hipTexRefGetFormat_Negative) {
  CHECK_IMAGE_SUPPORT
  hipModule_t module = nullptr;
  hipTexRef tex_ref = nullptr;
  int num_channels = 1;
  hipArray_Format format_get;
  hipArray_Format format_set = HIP_AD_FORMAT_UNSIGNED_INT32;

  HIP_CHECK(hipFree(nullptr));
  HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
  HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));

  HIP_CHECK(hipTexRefSetFormat(tex_ref, format_set, num_channels));

#if HT_AMD
  HIP_CHECK_ERROR(hipTexRefGetFormat(&format_get, &num_channels, nullptr), hipErrorInvalidValue);
#else
  HIP_CHECK_ERROR(hipTexRefGetFormat(&format_get, &num_channels, nullptr),
                  hipErrorInvalidResourceHandle);
#endif

  HIP_CHECK(hipModuleUnload(module));
}

#endif  // __HIP_PLATFORM_AMD__ || CUDA_VERSION < CUDA_12000
