/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>

#if defined(__HIP_PLATFORM_AMD__) || CUDA_VERSION < CUDA_12000

texture<float, 1, hipReadModeElementType> tex;

/**
 * Test Description
 * ------------------------
 *    - Positive test for hipGetTextureAlignmentOffset
 *    - Offset should always be 0
 * Test source
 * ------------------------
 *    - unit/texture/hipGetTextureAlignmentOffset.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
TEST_CASE("Unit_hipGetTextureAlignmentOffset_Positive") {
  CHECK_IMAGE_SUPPORT

  size_t offset = 0;
  size_t* tex_buf;
  hipChannelFormatDesc chanDesc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);

  HIP_CHECK(hipMalloc(&tex_buf, 32));
  HIP_CHECK(hipBindTexture(&offset, tex, reinterpret_cast<void*>(tex_buf), chanDesc, 32));
  HIP_CHECK(hipGetTextureAlignmentOffset(&offset, &tex));
  REQUIRE(offset == 0);

  HIP_CHECK(hipFree(tex_buf));
  HIP_CHECK(hipUnbindTexture(tex));
}

/**
 * Test Description
 * ------------------------
 *    - Negative test for hipGetTextureAlignmentOffset
 *    - Test should give invalid error if one of params is NULL
 * Test source
 * ------------------------
 *    - unit/texture/hipGetTextureAlignmentOffset.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
TEST_CASE("Unit_hipGetTextureAlignmentOffset_Negative") {
  CHECK_IMAGE_SUPPORT
  size_t offset = 0;
  size_t* tex_buf;
  hipChannelFormatDesc chanDesc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);

  HIP_CHECK(hipMalloc(&tex_buf, 32));
  HIP_CHECK(hipBindTexture(&offset, tex, reinterpret_cast<void*>(tex_buf), chanDesc, 32));

  SECTION("offset is nullptr") {
    HIP_CHECK_ERROR(hipGetTextureAlignmentOffset(nullptr, &tex), hipErrorInvalidValue);
  }

  SECTION("texture is nullptr") {
    HIP_CHECK_ERROR(hipGetTextureAlignmentOffset(&offset, nullptr), hipErrorInvalidTexture);
  }

  HIP_CHECK(hipFree(tex_buf));
  HIP_CHECK(hipUnbindTexture(tex));
}

#endif
