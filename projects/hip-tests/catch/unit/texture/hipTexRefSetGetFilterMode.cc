/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#if defined(__HIP_PLATFORM_AMD__) || CUDA_VERSION < CUDA_12000
texture<float, 2, hipReadModeElementType> tex;
// Test for hipTexRefSetFilterMode and hipTexRefGetFilterMode, including error handling
HIP_TEST_CASE(Unit_hipTexRefSetGetFilterMode) {
  CHECK_IMAGE_SUPPORT;

  // Retrieve the texture reference for our symbol
  const textureReference* texRefConst = nullptr;
  HIP_CHECK(hipGetTextureReference(&texRefConst, &tex));
  REQUIRE(texRefConst != nullptr);
  // Implementation expects non-const textureReference*
  textureReference* texRef = const_cast<textureReference*>(texRefConst);

  hipTextureFilterMode mode;

  SECTION("Default filter mode is Point") {
    HIP_CHECK(hipTexRefGetFilterMode(&mode, texRef));
    REQUIRE(mode == hipFilterModePoint);
  }

  SECTION("Set filter mode to Linear and verify") {
    HIP_CHECK(hipTexRefSetFilterMode(texRef, hipFilterModeLinear));
    HIP_CHECK(hipTexRefGetFilterMode(&mode, texRef));
    REQUIRE(mode == hipFilterModeLinear);
  }

  SECTION("Set filter mode back to Point and verify") {
    HIP_CHECK(hipTexRefSetFilterMode(texRef, hipFilterModePoint));
    HIP_CHECK(hipTexRefGetFilterMode(&mode, texRef));
    REQUIRE(mode == hipFilterModePoint);
  }

  SECTION("Invalid arguments: null texture reference pointer") {
    // Setting filter mode with null texRef should fail
    hipError_t errSet = hipTexRefSetFilterMode(nullptr, hipFilterModeLinear);
    REQUIRE(errSet == hipErrorInvalidValue);

    // Getting filter mode with null texRef should fail
    hipError_t errGetRef = hipTexRefGetFilterMode(&mode, nullptr);
    REQUIRE(errGetRef == hipErrorInvalidValue);

    // Getting filter mode with null mode pointer should fail
    hipError_t errGetMode = hipTexRefGetFilterMode(nullptr, texRef);
    REQUIRE(errGetMode == hipErrorInvalidValue);
  }
}
#endif
