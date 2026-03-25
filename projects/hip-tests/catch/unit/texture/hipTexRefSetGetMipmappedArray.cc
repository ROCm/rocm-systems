/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#if defined(__HIP_PLATFORM_AMD__) || CUDA_VERSION < CUDA_12000
texture<float, 2, hipReadModeElementType> tex;

// Test for hipTexRefSetMipmappedArray and hipTexRefGetMipmappedArray, including error handling
HIP_TEST_CASE(Unit_hipTexRefSetGetMipmappedArray) {
  CHECK_IMAGE_SUPPORT;

  // Retrieve the texture reference for our symbol
  const textureReference* texRefConst = nullptr;
  HIP_CHECK(hipGetTextureReference(&texRefConst, &tex));
  REQUIRE(texRefConst != nullptr);
  // Implementation expects non-const textureReference*
  textureReference* texRef = const_cast<textureReference*>(texRefConst);
  hipMipmappedArray_t mipArr = nullptr;
  hipMipmappedArray_t outArr = nullptr;
  unsigned int Flags = 0;


  SECTION("Default mipmapped array GET returns invalid value when none bound") {
    hipError_t err = hipTexRefGetMipMappedArray(&outArr, texRef);
    REQUIRE(err == hipErrorInvalidValue);
  }

  SECTION("Set and get mipmapped array") {
    hipMipmappedArray_t mipmapped_array;
    HIP_RESOURCE_DESC res_desc{};
    hipModule_t module = nullptr;
    hipExtent extent;
    hipChannelFormatDesc channel_desc;
    unsigned int width = 256, height = 256, mipmap_level = 2;

    res_desc.resType = HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY;

    channel_desc = hipCreateChannelDesc<float>();
    extent = make_hipExtent(width, height, 0);
    auto res = hipMallocMipmappedArray(&mipmapped_array, &channel_desc, extent, 2 * mipmap_level,
                                       hipArrayDefault);
    if (res == hipErrorNotSupported) {
      SUCCEED("Mipmapped arrays not supported on this device");
      return;
    }
    HIP_CHECK(res);
    HIP_CHECK(hipFree(nullptr));
    HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
    HIP_CHECK(hipModuleGetTexRef(&texRef, module, "tex"));
    HIP_CHECK(hipTexRefSetFlags(texRef, HIP_TRSF_NORMALIZED_COORDINATES));
    HIP_CHECK(hipTexRefSetMipmappedArray(texRef, mipmapped_array, HIP_TRSA_OVERRIDE_FORMAT));
    HIP_CHECK(hipTexRefGetMipMappedArray(&outArr, texRef));
    REQUIRE(outArr == mipmapped_array);
    HIP_CHECK(hipFreeMipmappedArray(mipmapped_array));
  }

  SECTION("Invalid arguments: null pointers") {
    hipError_t err;
    err = hipTexRefSetMipmappedArray(nullptr, mipArr, Flags);
    REQUIRE(err == hipErrorInvalidValue);
    err = hipTexRefGetMipMappedArray(&outArr, nullptr);
    REQUIRE(err == hipErrorInvalidValue);
    err = hipTexRefGetMipMappedArray(nullptr, texRef);
    REQUIRE(err == hipErrorInvalidValue);
  }
}
#endif
