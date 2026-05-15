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
// Test for hipTexRefSetMipmapFilterMode and hipTexRefGetMipmapFilterMode, including error handling
TEST_CASE("Unit_hipTexRefSetGetMipmapFilterMode") {
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
