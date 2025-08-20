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

#include <hip_test_common.hh>

/**
 * @addtogroup hipMallocMipmappedArray hipMallocMipmappedArray
 * @{
 * @ingroup TextureTest
 */

/**
 * Test Description
 * ------------------------
 *    - Negative parameters test for `hipMallocMipmappedArray`.
 * Test source
 * ------------------------
 *    - unit/texture/hipMallocMipmappedArray.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
TEST_CASE("Unit_hipMallocMipmappedArray_Negative_Parameters") {
  CHECK_IMAGE_SUPPORT;

#ifdef __linux__
  HipTest::HIP_SKIP_TEST("Mipmap APIs are not supported on Linux");
  return;
#endif  //__linux__

  hipMipmappedArray_t array;
  hipChannelFormatDesc desc = hipCreateChannelDesc<float>();
  hipExtent extent = make_hipExtent(4, 4, 6);
  unsigned int levels = 1 + std::log2(extent.depth);

  SECTION("mipmappedArray is nullptr") {
    HIP_CHECK_ERROR(hipMallocMipmappedArray(nullptr, &desc, extent, levels, 0),
                    hipErrorInvalidValue);
  }

  SECTION("desc is nullptr") {
    HIP_CHECK_ERROR(hipMallocMipmappedArray(&array, nullptr, extent, levels, 0),
                    hipErrorInvalidValue);
  }

  SECTION("extent is zero") {
    extent = {};
    HIP_CHECK_ERROR(hipMallocMipmappedArray(&array, &desc, extent, levels, 0),
                    hipErrorInvalidValue);
  }

  SECTION("invalid flags") {
    HIP_CHECK_ERROR(
        hipMallocMipmappedArray(&array, &desc, extent, levels, static_cast<unsigned int>(-1)),
        hipErrorInvalidValue);
  }

  SECTION("hipArrayCubemap && depth != height") {
    extent.height = 5;
    HIP_CHECK_ERROR(hipMallocMipmappedArray(&array, &desc, extent, levels, hipArrayCubemap),
                    hipErrorInvalidValue);
  }

  SECTION("hipArrayCubemap && depth != 6") {
    extent.depth = 12;
    HIP_CHECK_ERROR(hipMallocMipmappedArray(&array, &desc, extent, levels, hipArrayCubemap),
                    hipErrorInvalidValue);
  }

  SECTION("hipArrayCubemap && hipArrayLayered && depth is not a multiple of 6") {
    extent.depth = 13;
    HIP_CHECK_ERROR(
        hipMallocMipmappedArray(&array, &desc, extent, levels, hipArrayCubemap | hipArrayLayered),
        hipErrorInvalidValue);
  }

  SECTION("hipArrayTextureGather && 1D array") {
    extent.height = 0;
    extent.depth = 0;
    HIP_CHECK_ERROR(hipMallocMipmappedArray(&array, &desc, extent, levels, hipArrayTextureGather),
                    hipErrorInvalidValue);
  }

  SECTION("hipArrayTextureGather && 3D array") {
    HIP_CHECK_ERROR(hipMallocMipmappedArray(&array, &desc, extent, levels, hipArrayTextureGather),
                    hipErrorInvalidValue);
  }

#if HT_NVIDIA  // Disabled due to defect EXSWHTEC-365
  SECTION("hipArraySparse && 1D array") {
    extent.height = 0;
    extent.depth = 0;
    HIP_CHECK_ERROR(hipMallocMipmappedArray(&array, &desc, extent, levels, cudaArraySparse),
                    hipErrorInvalidValue);
  }

  SECTION("hipArraySparse && cubemap array") {
    HIP_CHECK_ERROR(
        hipMallocMipmappedArray(&array, &desc, extent, levels, hipArrayCubemap | cudaArraySparse),
        hipErrorInvalidValue);
  }
#endif
}

/**
 * End doxygen group TextureTest.
 * @}
 */
