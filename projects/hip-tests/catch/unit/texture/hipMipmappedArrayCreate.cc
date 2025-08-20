/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip_array_common.hh>
#include <hip_test_common.hh>

/**
 * @addtogroup hipMipmappedArrayCreate hipMipmappedArrayCreate
 * @{
 * @ingroup TextureTest
 */

/**
 * Test Description
 * ------------------------
 *    - Negative parameters test for `hipMipmappedArrayCreate`.
 * Test source
 * ------------------------
 *    - unit/texture/hipMipmappedArrayCreate.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
TEST_CASE("Unit_hipMipmappedArrayCreate_Negative_Parameters") {
  CHECK_IMAGE_SUPPORT;

#ifdef __linux__
  HipTest::HIP_SKIP_TEST("Mipmap APIs are not supported on Linux");
  return;
#endif  //__linux__

  hipmipmappedArray array;

  HIP_ARRAY3D_DESCRIPTOR desc = {};
  using vec_info = vector_info<float>;
  desc.Format = vec_info::format;
  desc.NumChannels = vec_info::size;
  desc.Width = 4;
  desc.Height = 4;
  desc.Depth = 6;
  desc.Flags = 0;

  unsigned int levels = 1 + std::log2(desc.Depth);

  HIP_CHECK(hipFree(0));

  SECTION("mipmappedArray is nullptr") {
    HIP_CHECK_ERROR(hipMipmappedArrayCreate(nullptr, &desc, levels), hipErrorInvalidValue);
  }

  SECTION("desc is nullptr") {
    HIP_CHECK_ERROR(hipMipmappedArrayCreate(&array, nullptr, levels), hipErrorInvalidValue);
  }

  SECTION("extent is zero") {
    desc.Width = 0;
    desc.Height = 0;
    desc.Depth = 0;
    HIP_CHECK_ERROR(hipMipmappedArrayCreate(&array, &desc, levels), hipErrorInvalidValue);
  }

  SECTION("invalid flags") {
    desc.Flags = static_cast<unsigned int>(-1);
    HIP_CHECK_ERROR(hipMipmappedArrayCreate(&array, &desc, levels), hipErrorInvalidValue);
  }

  SECTION("hipArrayCubemap && depth != 6") {
    desc.Depth = 5;
    desc.Flags = hipArrayCubemap;
    HIP_CHECK_ERROR(hipMipmappedArrayCreate(&array, &desc, levels), hipErrorInvalidValue);
  }

  SECTION("hipArrayCubemap && hipArrayLayered && depth is not a multiple of 6") {
    desc.Depth = 13;
    desc.Flags = hipArrayCubemap | hipArrayLayered;
    HIP_CHECK_ERROR(hipMipmappedArrayCreate(&array, &desc, levels), hipErrorInvalidValue);
  }

  SECTION("hipArrayTextureGather && 1D array") {
    desc.Height = 0;
    desc.Depth = 0;
    desc.Flags = hipArrayTextureGather;
    HIP_CHECK_ERROR(hipMipmappedArrayCreate(&array, &desc, levels), hipErrorInvalidValue);
  }

  SECTION("hipArrayTextureGather && 3D array") {
    desc.Flags = hipArrayTextureGather;
    HIP_CHECK_ERROR(hipMipmappedArrayCreate(&array, &desc, levels), hipErrorInvalidValue);
  }

#if HT_NVIDIA  // Disabled due to defect EXSWHTEC-365
  SECTION("hipArraySparse && 1D array") {
    desc.Height = 0;
    desc.Depth = 0;
    desc.Flags = cudaArraySparse;
    HIP_CHECK_ERROR(hipMipmappedArrayCreate(&array, &desc, levels), hipErrorInvalidValue);
  }

  SECTION("hipArraySparse && cubemap array") {
    desc.Flags = hipArrayCubemap | cudaArraySparse;
    HIP_CHECK_ERROR(hipMipmappedArrayCreate(&array, &desc, levels), hipErrorInvalidValue);
  }
#endif
}

/**
 * End doxygen group TextureTest.
 * @}
 */
