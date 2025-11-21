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
// Test for hipTexRefSetMipmapLevelBias and hipTexRefGetMipmapLevelBias, including error handling
TEST_CASE("Unit_hipTexRefSetGetMipmapLevelBias") {
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
