/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#if defined(__HIP_PLATFORM_AMD__) || CUDA_VERSION < CUDA_12000
texture<float, 2, hipReadModeElementType> tex;
// Test for hipTexRefSetMipmapFilterMode and hipTexRefGetMipmapFilterMode, including error handling
TEST_CASE(Unit_hipTexRefSetGetMipmapFilterMode) {
  CHECK_IMAGE_SUPPORT;

  // Retrieve the texture reference for our symbol
  const textureReference* texRefConst = nullptr;
  HIP_CHECK(hipGetTextureReference(&texRefConst, &tex));
  REQUIRE(texRefConst != nullptr);
  // Implementation expects non-const textureReference*
  textureReference* texRef = const_cast<textureReference*>(texRefConst);

  hipTextureFilterMode mipMode;

  SECTION("Set mipmap filter mode to Linear and verify") {
    HIP_CHECK(hipTexRefSetMipmapFilterMode(texRef, hipFilterModeLinear));
    auto res = hipTexRefGetMipmapFilterMode(&mipMode, texRef);
    REQUIRE(res == hipErrorInvalidValue);
    REQUIRE(mipMode == hipFilterModeLinear);
  }

  SECTION("Set mipmap filter mode back to Point and verify") {
    HIP_CHECK(hipTexRefSetMipmapFilterMode(texRef, hipFilterModePoint));
    auto res = hipTexRefGetMipmapFilterMode(&mipMode, texRef);
    REQUIRE(res == hipErrorInvalidValue);
    REQUIRE(mipMode == hipFilterModePoint);
  }

  SECTION("Invalid arguments: null pointers") {
    hipError_t err;
    err = hipTexRefSetMipmapFilterMode(nullptr, hipFilterModeLinear);
    REQUIRE(err == hipErrorInvalidValue);
    err = hipTexRefGetMipmapFilterMode(&mipMode, nullptr);
    REQUIRE(err == hipErrorInvalidValue);
    err = hipTexRefGetMipmapFilterMode(nullptr, texRef);
    REQUIRE(err == hipErrorInvalidValue);
  }
}
#endif
