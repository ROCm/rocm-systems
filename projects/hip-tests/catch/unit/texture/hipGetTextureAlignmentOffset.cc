/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#if defined(__HIP_PLATFORM_AMD__) || CUDA_VERSION < CUDA_12000

texture<float, 1, hipReadModeElementType> tex;

/**
 * Test Description
 * ------------------------
 *    - Positive test for hipGetTextureAlignmentOffset
 *    - Offset should always be 0
 * Test source
 * ------------------------
 *    - unit/texture/hipGetTextureAlignmentOffset.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
HIP_TEST_CASE(Unit_hipGetTextureAlignmentOffset_Positive) {
  CHECK_IMAGE_SUPPORT

  size_t offset = 0;
  size_t* tex_buf;
  hipChannelFormatDesc chanDesc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);

  HIP_CHECK(hipMalloc(&tex_buf, 32));
  HIP_CHECK(hipBindTexture(&offset, tex, reinterpret_cast<void*>(tex_buf), chanDesc, 32));
  HIP_CHECK(hipGetTextureAlignmentOffset(&offset, &tex));
  REQUIRE(offset == 0);

  HIP_CHECK(hipFree(tex_buf));
  HIP_CHECK(hipUnbindTexture(tex));
}

/**
 * Test Description
 * ------------------------
 *    - Negative test for hipGetTextureAlignmentOffset
 *    - Test should give invalid error if one of params is NULL
 * Test source
 * ------------------------
 *    - unit/texture/hipGetTextureAlignmentOffset.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
HIP_TEST_CASE(Unit_hipGetTextureAlignmentOffset_Negative) {
  CHECK_IMAGE_SUPPORT
  size_t offset = 0;
  size_t* tex_buf;
  hipChannelFormatDesc chanDesc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);

  HIP_CHECK(hipMalloc(&tex_buf, 32));
  HIP_CHECK(hipBindTexture(&offset, tex, reinterpret_cast<void*>(tex_buf), chanDesc, 32));

  SECTION("offset is nullptr") {
    HIP_CHECK_ERROR(hipGetTextureAlignmentOffset(nullptr, &tex), hipErrorInvalidValue);
  }

  SECTION("texture is nullptr") {
    HIP_CHECK_ERROR(hipGetTextureAlignmentOffset(&offset, nullptr), hipErrorInvalidTexture);
  }

  HIP_CHECK(hipFree(tex_buf));
  HIP_CHECK(hipUnbindTexture(tex));
}

#endif
