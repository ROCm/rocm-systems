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
 * @addtogroup hipArrayGetInfo hipArrayGetInfo
 * @{
 * @ingroup MemoryTest
 * `hipArrayGetInfo(hipChannelFormatDesc* desc, hipExtent* extent, unsigned int* flags, hipArray*
 * array)` - Gets info about the specified array.
 */

#include <hip_test_common.hh>
#include <resource_guards.hh>

/**
 * Test Description
 * ------------------------
 *  - Basic sanity test for `hipArrayGetInfo`.
 * Test source
 * ------------------------
 *  - unit/memory/hipArrayGetInfo.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.7
 */
TEST_CASE("Unit_hipArrayGetInfo_Positive_Basic") {
  CHECK_IMAGE_SUPPORT

  ArrayAllocGuard<float> array(make_hipExtent(1024, 4, 2));

  hipChannelFormatDesc desc;
  hipExtent extent;
  unsigned int flags = 1;

  HIP_CHECK(hipArrayGetInfo(&desc, &extent, &flags, array.ptr()));

  REQUIRE(extent.width == 1024);
  REQUIRE(extent.height == 4);
  REQUIRE(extent.depth == 2);

  REQUIRE(flags == 0);

  auto expected_desc = hipCreateChannelDesc<float>();
  REQUIRE(desc.x == expected_desc.x);
  REQUIRE(desc.y == expected_desc.y);
  REQUIRE(desc.z == expected_desc.z);
  REQUIRE(desc.w == expected_desc.w);
  REQUIRE(desc.f == expected_desc.f);
}

/**
 * Test Description
 * ------------------------
 *  - Negative parameters test for `hipArrayGetInfo`.
 * Test source
 * ------------------------
 *  - unit/memory/hipArrayGetInfo.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.7
 */
TEST_CASE("Unit_hipArrayGetInfo_Negative_Parameters") {
  CHECK_IMAGE_SUPPORT
  ArrayAllocGuard<float> array(make_hipExtent(1024, 4, 4));

  hipChannelFormatDesc desc;
  hipExtent extent;
  unsigned int flags;

  SECTION("array is nullptr") {
    HIP_CHECK_ERROR(hipArrayGetInfo(&desc, &extent, &flags, nullptr), hipErrorInvalidHandle);
  }

  SECTION("array is freed") {
    HIP_CHECK(hipFreeArray(array.ptr()));
    HIP_CHECK_ERROR(hipArrayGetInfo(&desc, &extent, &flags, array.ptr()), hipErrorInvalidHandle);
  }
}

/**
 * End doxygen group MemoryTest.
 * @}
 */
