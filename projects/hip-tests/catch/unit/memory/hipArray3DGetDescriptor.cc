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

/**
 * @addtogroup hipArray3DGetDescriptor hipArray3DGetDescriptor
 * @{
 * @ingroup MemoryTest
 * `hipArray3DGetDescriptor(HIP_ARRAY3D_DESCRIPTOR* pArrayDescriptor, hipArray* array)` -
 * Gets a 3D array descriptor.
 */

#include <hip_test_common.hh>
#include <resource_guards.hh>

/**
 * Test Description
 * ------------------------
 *  - Basic sanity test for `hipArray3DGetDescriptor`.
 * Test source
 * ------------------------
 *  - unit/memory/hipArray3DGetDescriptor.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.7
 */
TEST_CASE("Unit_hipArray3DGetDescriptor_Positive_Basic") {
  CHECK_IMAGE_SUPPORT
  DrvArrayAllocGuard<float> array(make_hipExtent(1024, 4, 2));

  HIP_ARRAY3D_DESCRIPTOR desc;
  HIP_CHECK(hipArray3DGetDescriptor(&desc, array.ptr()));

  using vec_info = vector_info<float>;
  REQUIRE(desc.Format == vec_info::format);
  REQUIRE(desc.NumChannels == vec_info::size);
  REQUIRE(desc.Width == 1024 / sizeof(float));
  REQUIRE(desc.Height == 4);
  REQUIRE(desc.Depth == 2);
  REQUIRE(desc.Flags == 0);
}

/**
 * Test Description
 * ------------------------
 *  - Negative parameters test for `hipArray3DGetDescriptor`.
 * Test source
 * ------------------------
 *  - unit/memory/hipArray3DGetDescriptor.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.7
 */
TEST_CASE("Unit_hipArray3DGetDescriptor_Negative_Parameters") {
  CHECK_IMAGE_SUPPORT
  DrvArrayAllocGuard<float> array(make_hipExtent(1024, 4, 2));

  HIP_ARRAY3D_DESCRIPTOR desc;

  SECTION("desc is nullptr") {
    HIP_CHECK_ERROR(hipArray3DGetDescriptor(nullptr, array.ptr()), hipErrorInvalidValue);
  }

  SECTION("array is nullptr") {
    HIP_CHECK_ERROR(hipArray3DGetDescriptor(&desc, nullptr), hipErrorInvalidHandle);
  }

  SECTION("array is freed") {
    HIP_CHECK(hipArrayDestroy(array.ptr()));
    HIP_CHECK_ERROR(hipArray3DGetDescriptor(&desc, array.ptr()), hipErrorInvalidHandle);
  }
}

/**
 * End doxygen group MemoryTest.
 * @}
 */
