/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>


#if defined(__HIP_PLATFORM_AMD__) || CUDA_VERSION < CUDA_12000

texture<float, 1, hipReadModeElementType> tex;

// It will be added for HIP in ROCm7.0
#if HT_NVIDIA
TEST_CASE(Unit_hipTexRefGetBorderColor_Positive) {
  CHECK_IMAGE_SUPPORT
  float set_border_color[3] = {1, 2, 3};
  float get_border_color[3] = {0, 0, 0};
  hipModule_t module = nullptr;
  hipTexRef tex_ref = nullptr;

  HIP_CHECK(hipFree(nullptr));
  HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
  HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));

  HIP_CHECK(hipTexRefSetBorderColor(tex_ref, set_border_color));
  HIP_CHECK(hipTexRefGetBorderColor(get_border_color, tex_ref));

  for (int i = 0; i < 3; i++) {
    REQUIRE(set_border_color[i] == get_border_color[i]);
  }
  HIP_CHECK(hipModuleUnload(module));
}
#endif

TEST_CASE(Unit_hipTexRefGetBorderColor_Negative) {
  CHECK_IMAGE_SUPPORT
  float border_color[3] = {0, 0, 0};
  hipModule_t module = nullptr;
  hipTexRef tex_ref = nullptr;

  HIP_CHECK(hipFree(nullptr));
  HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
  HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));

  SECTION("border_color is null") {
    HIP_CHECK_ERROR(hipTexRefGetBorderColor(nullptr, tex_ref), hipErrorInvalidValue);
  }

  SECTION("texture reference is null") {
#if HT_AMD
    HIP_CHECK_ERROR(hipTexRefGetBorderColor(border_color, nullptr), hipErrorInvalidValue);
#else
    HIP_CHECK_ERROR(hipTexRefGetBorderColor(border_color, nullptr), hipErrorInvalidResourceHandle);
#endif
  }

  HIP_CHECK(hipModuleUnload(module));
}

#endif
