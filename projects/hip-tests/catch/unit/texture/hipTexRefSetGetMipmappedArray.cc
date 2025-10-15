/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <hip_test_common.hh>
#if defined(__HIP_PLATFORM_AMD__) || CUDA_VERSION < CUDA_12000
texture<float, 2, hipReadModeElementType> tex;

// Test for hipTexRefSetMipmappedArray and hipTexRefGetMipmappedArray, including error handling
TEST_CASE("Unit_hipTexRefSetGetMipmappedArray") {
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

    HIP_CHECK(hipTexRefSetMipmappedArray(texRef, mipmapped_array, Flags));
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
