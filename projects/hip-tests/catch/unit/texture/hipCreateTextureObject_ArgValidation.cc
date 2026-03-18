/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

/**
 * @addtogroup hipCreateTextureObject hipCreateTextureObject
 * @{
 * @ingroup TextureTest
 */

#define N 512

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments for [hipCreateTextureObject](@ref
 * hipCreateTextureObject):
 *    -# When output pointer to the texture object is `nullptr`
 *      - Expected output: do not return `hipSuccess`
 *    -# When resource descriptor is `nullptr`
 *      - Expected output: do not return `hipSuccess`
 *    -# When texture descriptor is `nullptr`
 *      - Expected output: do not return `hipSuccess`
 *  - Validates handling of invalid arguments for [hipDestroyTextureObject](@ref
 * hipDestroyTextureObject):
 *    -# When texture object handle is `nullptr`
 *      - Expected output: return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/texture/hipCreateTextureObject_ArgValidation.cc
 * Test requirements
 * ------------------------
 *  - Textures supported on device
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipCreateTextureObject_ArgValidation) {
  CHECK_IMAGE_SUPPORT

  float* texBuf;
  hipError_t ret;
  constexpr int xsize = 32;
  hipResourceDesc resDesc;
  hipTextureDesc texDesc;
  hipTextureObject_t texObj;

  // Initialization
  HIP_CHECK(hipMalloc(&texBuf, N * sizeof(float)));
  // Populate resource descriptor
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = hipResourceTypeLinear;
  resDesc.res.linear.devPtr = texBuf;
  resDesc.res.linear.desc = hipCreateChannelDesc(xsize, 0, 0, 0, hipChannelFormatKindFloat);
  resDesc.res.linear.sizeInBytes = N * sizeof(float);

  // Populate texture descriptor
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = hipReadModeElementType;


  // Sections
  SECTION("TextureObject as nullptr") {
    ret = hipCreateTextureObject(nullptr, &resDesc, &texDesc, nullptr);
    REQUIRE(ret != hipSuccess);
  }

  SECTION("Resouce Descriptor as nullptr") {
    ret = hipCreateTextureObject(&texObj, nullptr, &texDesc, nullptr);
    REQUIRE(ret != hipSuccess);
  }

  SECTION("Texture Descriptor as nullptr") {
    if ((TestContext::get()).isAmd()) {
      ret = hipCreateTextureObject(&texObj, &resDesc, nullptr, nullptr);
      REQUIRE(ret != hipSuccess);
    } else {
      // API expected to return failure. Test skipped
      // on nvidia as api returns success and would lead
      // to unexpected behavior with app.
      WARN("Texture Desc(nullptr) skipped on nvidia");
    }
  }

  SECTION("Destroy TextureObject with nullptr") {
    ret = hipDestroyTextureObject((hipTextureObject_t) nullptr);
    // api to return success and no crash seen.
    REQUIRE(ret == hipSuccess);
  }

  // De-Initialization
  HIP_CHECK(hipFree(texBuf));
}

/**
 * End doxygen group TextureTest.
 * @}
 */
