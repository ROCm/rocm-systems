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
#include <float.h>
#if defined(__HIP_PLATFORM_AMD__) || CUDA_VERSION < CUDA_12000
texture<float, 2, hipReadModeElementType> tex;
// Test for hipTexRefSetMipmapLevelClamp and hipTexRefGetMipmapLevelClamp, including error handling
TEST_CASE("Unit_texRefSetGetMipmapLevelClamp") {
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
