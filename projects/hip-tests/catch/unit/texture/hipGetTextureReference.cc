/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#if defined(__HIP_PLATFORM_AMD__) || CUDA_VERSION < CUDA_12000

texture<float, 2, hipReadModeElementType> tex;

HIP_TEST_CASE(Unit_hipGetTextureReference_Positive) {
  CHECK_IMAGE_SUPPORT

  const textureReference* tex_ref = nullptr;
  HIP_CHECK(hipGetTextureReference(&tex_ref, &tex));
  REQUIRE(tex_ref != nullptr);
}

HIP_TEST_CASE(Unit_hipGetTextureReference_Negative) {
  CHECK_IMAGE_SUPPORT

  const textureReference* tex_ref = nullptr;

  // Cuda crashes with SIGSEGV
#if HT_AMD
  SECTION("texture reference is null") {
    HIP_CHECK_ERROR(hipGetTextureReference(nullptr, &tex), hipErrorInvalidValue);
  }
#endif

  SECTION("texture is null") {
#if HT_AMD
    HIP_CHECK(hipGetTextureReference(&tex_ref, nullptr));
#else
    HIP_CHECK_ERROR(hipGetTextureReference(&tex_ref, nullptr), hipErrorInvalidTexture);
#endif
  }
}

#endif
