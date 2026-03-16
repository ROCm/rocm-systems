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

/**
 * Test Description
 * ------------------------
 *  - Validates handling of a regular `nullptr` array
 *    - Expected output: do not return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/texture/hipCreateTextureObject_Array.cc
 * Test requirements
 * ------------------------
 *  - Textures supported on device
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipCreateTextureObject_ArrayResource) {
  CHECK_IMAGE_SUPPORT

  hipError_t ret;
  hipResourceDesc resDesc;
  hipTextureDesc texDesc;
  hipTextureObject_t texObj;

  /* set resource type as hipResourceTypeArray and array(nullptr) */
  // Populate resource descriptor
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = hipResourceTypeArray;
  resDesc.res.array.array = nullptr;

  // Populate texture descriptor
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = hipReadModeElementType;

  ret = hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
  REQUIRE(ret != hipSuccess);
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of a regular `nullptr` mipmapped array
 *    - Expected output: do not return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/texture/hipCreateTextureObject_Array.cc
 * Test requirements
 * ------------------------
 *  - Textures supported on device
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipCreateTextureObject_MmArrayResource) {
  CHECK_IMAGE_SUPPORT

  hipError_t ret;
  hipResourceDesc resDesc;
  hipTextureDesc texDesc;
  hipTextureObject_t texObj;

  /* set resource type as hipResourceTypeMipmappedArray and mipmap(nullptr) */
  // Populate resource descriptor
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = hipResourceTypeMipmappedArray;
  resDesc.res.mipmap.mipmap = nullptr;

  // Populate texture descriptor
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = hipReadModeElementType;

  ret = hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
  REQUIRE(ret != hipSuccess);
}

/**
 * End doxygen group TextureTest.
 * @}
 */
