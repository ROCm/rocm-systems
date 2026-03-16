/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#if defined(__HIP_PLATFORM_AMD__) || CUDA_VERSION < CUDA_12000
texture<float, 2, hipReadModeElementType> tex;
// Test for hipTexRefSetMipmapLevelBias and hipTexRefGetMipmapLevelBias, including error handling
TEST_CASE(Unit_hipTexRefSetGetMipmapLevelBias) {
  CHECK_IMAGE_SUPPORT;

  // Retrieve the texture reference for our symbol
  const textureReference* texRefConst = nullptr;
  HIP_CHECK(hipGetTextureReference(&texRefConst, &tex));
  REQUIRE(texRefConst != nullptr);
  // Implementation expects non-const textureReference*
  textureReference* texRef = const_cast<textureReference*>(texRefConst);

  float bias = 0.0;

  SECTION("Set mipmap level bias to custom value and verify") {
    float newBias = 2.25;
    HIP_CHECK(hipTexRefSetMipmapLevelBias(texRef, newBias));
    auto res = hipTexRefGetMipmapLevelBias(&bias, texRef);
    REQUIRE(res == hipErrorInvalidValue);
    REQUIRE(bias == newBias);
  }

  SECTION("Invalid arguments: null pointers") {
    hipError_t err;
    err = hipTexRefSetMipmapLevelBias(nullptr, 1.0f);
    REQUIRE(err == hipErrorInvalidValue);
    err = hipTexRefGetMipmapLevelBias(nullptr, texRef);
    REQUIRE(err == hipErrorInvalidValue);
    err = hipTexRefGetMipmapLevelBias(&bias, nullptr);
    REQUIRE(err == hipErrorInvalidValue);
  }
}
#endif
