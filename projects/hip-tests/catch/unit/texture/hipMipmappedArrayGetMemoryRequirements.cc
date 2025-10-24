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
 * @addtogroup hipMipmappedArrayGetMemoryRequirements hipMipmappedArrayGetMemoryRequirements 
 * @{
 * @ingroup TextureTest
 */

/**
 * Test Description
 * ------------------------
 * - This test will create mipmapped array for different data types and check
 * how much memory allocated using hipMipmappedArrayGetMemoryRequirements Api
 * source
 * ------------------------
 *    - unit/texture/hipMipmappedArrayGetMemoryRequirements.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.2
 */
TEMPLATE_TEST_CASE("Unit_hipMipmappedArrayGetMemoryRequirements_positive", "",
                   uint, int, int4, ushort, short2, char, uchar2, char4, float,
                   float2, float4) {
  CHECK_IMAGE_SUPPORT;

#ifdef __linux__
  HipTest::HIP_SKIP_TEST("Mipmap APIs are not supported on Linux");
  return;
#endif  // __linux__

  const int device_id = 0;
  hipArrayMemoryRequirements memoryRequirements{};
  hipmipmappedArray array;
  HIP_ARRAY3D_DESCRIPTOR desc = {};
  using vec_info = vector_info<TestType>;
  desc.Format = vec_info::format;
  desc.NumChannels = vec_info::size;
  desc.Width = 256;
  desc.Height = 256;
  desc.Depth = 1;
  desc.Flags = 0;

  unsigned int levels = 1 + std::log2(desc.Depth);

  HIP_CHECK(hipFree(0));
  HIP_CHECK(hipMipmappedArrayCreate(&array, &desc, levels));

  HIP_CHECK(hipMipmappedArrayGetMemoryRequirements(&memoryRequirements, array,
                                                   device_id));
  REQUIRE(memoryRequirements.size ==
          desc.Width * desc.Height * desc.Depth * sizeof(TestType));
  HIP_CHECK(hipMipmappedArrayDestroy(array));
}

/**
 * Test Description
 * ------------------------
 * - This test will verify the behavior of hipMipmappedArrayGetMemoryRequirements api with invalid array
 * source
 * ------------------------
 *    - unit/texture/hipMipmappedArrayGetMemoryRequirements.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.2
 */
TEST_CASE("Unit_hipMipmappedArrayGetMemoryRequirements_Negative_Parameters") {
  CHECK_IMAGE_SUPPORT;

#ifdef __linux__
  HipTest::HIP_SKIP_TEST("Mipmap APIs are not supported on Linux");
  return;
#endif  //__linux__

  const int device_id = 0;
  hipArrayMemoryRequirements memoryRequirements{};
  hipmipmappedArray array;
    HIP_CHECK_ERROR(hipMipmappedArrayGetMemoryRequirements(&memoryRequirements, array, device_id), hipErrorInvalidValue);
}

/**
 * End doxygen group TextureTest.
 * @}
 */
