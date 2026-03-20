/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <float.h>
#if defined(__HIP_PLATFORM_AMD__) || CUDA_VERSION < CUDA_12000
texture<float, 2, hipReadModeElementType> tex;
// Test for hipTexRefSetMipmapLevelClamp and hipTexRefGetMipmapLevelClamp, including error handling
HIP_TEST_CASE(Unit_texRefSetGetMipmapLevelClamp) {
  CHECK_IMAGE_SUPPORT;

  // Retrieve the texture reference for our symbol
  const textureReference* texRefConst = nullptr;
  HIP_CHECK(hipGetTextureReference(&texRefConst, &tex));
  REQUIRE(texRefConst != nullptr);
  // Implementation expects non-const textureReference*
  textureReference* texRef = const_cast<textureReference*>(texRefConst);


  float minClamp = 0.0f, maxClamp = 0.0f;

  SECTION("Set mipmap level clamp to custom values and verify") {
    float newMin = 1.5f, newMax = 5.5f;
    HIP_CHECK(hipTexRefSetMipmapLevelClamp(texRef, newMin, newMax));
    auto res = hipTexRefGetMipmapLevelClamp(&minClamp, &maxClamp, texRefConst);
    REQUIRE(res == hipErrorInvalidValue);
    REQUIRE(minClamp == newMin);
    REQUIRE(maxClamp == newMax);
  }

  SECTION("Invalid arguments: null pointers") {
    hipError_t err;
    err = hipTexRefSetMipmapLevelClamp(nullptr, 1.0f, 2.0f);
    REQUIRE(err == hipErrorInvalidValue);
    err = hipTexRefGetMipmapLevelClamp(nullptr, &maxClamp, texRefConst);
    REQUIRE(err == hipErrorInvalidValue);
    err = hipTexRefGetMipmapLevelClamp(&minClamp, nullptr, texRefConst);
    REQUIRE(err == hipErrorInvalidValue);
    err = hipTexRefGetMipmapLevelClamp(&minClamp, &maxClamp, nullptr);
    REQUIRE(err == hipErrorInvalidValue);
  }
}
#endif
