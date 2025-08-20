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
 * @addtogroup hipCreateSurfaceObject hipCreateSurfaceObject
 * @{
 * @ingroup SurfaceTest
 */

/**
 * Test Description
 * ------------------------
 *    - Negative parameters test for `hipCreateSurfaceObject`.
 * Test source
 * ------------------------
 *    - unit/texture/hipCreateSurfaceObject.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
TEST_CASE("Unit_hipCreateSurfaceObject_Negative_Parameters") {
  CHECK_IMAGE_SUPPORT

  hipArray_t array;
  hipChannelFormatDesc desc = hipCreateChannelDesc<float>();

  HIP_CHECK(hipMallocArray(&array, &desc, 64, 0, hipArraySurfaceLoadStore));

  hipSurfaceObject_t surf;

  hipResourceDesc resc = {};
  resc.resType = hipResourceTypeArray;
  resc.res.array.array = array;

  SECTION("pSurfObject is nullptr") {
    HIP_CHECK_ERROR(hipCreateSurfaceObject(nullptr, &resc), hipErrorInvalidValue);
  }

  SECTION("pResDesc is nullptr") {
    HIP_CHECK_ERROR(hipCreateSurfaceObject(&surf, nullptr), hipErrorInvalidValue);
  }

  SECTION("invalid resource type") {
    resc.resType = hipResourceTypeLinear;
    HIP_CHECK_ERROR(hipCreateSurfaceObject(&surf, &resc), hipErrorInvalidValue);
  }

#if HT_NVIDIA  // DIsalbed due to defect EXSWHTEC-366
  SECTION("array handle is nullptr") {
    resc.res.array.array = nullptr;
    HIP_CHECK_ERROR(hipCreateSurfaceObject(&surf, &resc), hipErrorInvalidHandle);
  }
#endif

#if HT_NVIDIA  // Disalbed due to defect EXSWHTEC-367
  SECTION("freed array handle") {
    hipArray_t invalid_array;
    HIP_CHECK(hipMallocArray(&invalid_array, &desc, 64, 0, hipArraySurfaceLoadStore));
    HIP_CHECK(hipFreeArray(invalid_array));
    resc.res.array.array = invalid_array;
    HIP_CHECK_ERROR(hipCreateSurfaceObject(&surf, &resc), hipErrorContextIsDestroyed);
  }
#endif

  HIP_CHECK(hipFreeArray(array));
}

/**
 * End doxygen group SurfaceTest.
 * @}
 */
