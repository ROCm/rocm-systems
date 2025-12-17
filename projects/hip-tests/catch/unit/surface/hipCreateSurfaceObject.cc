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
    HIP_CHECK_ERROR(hipCreateSurfaceObject(nullptr, &resc),
                    hipErrorInvalidValue);
  }

  SECTION("pResDesc is nullptr") {
    HIP_CHECK_ERROR(hipCreateSurfaceObject(&surf, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("invalid resource type") {
    resc.resType = hipResourceTypeLinear;
    HIP_CHECK_ERROR(hipCreateSurfaceObject(&surf, &resc), hipErrorInvalidValue);
  }

#if HT_NVIDIA  // DIsalbed due to defect EXSWHTEC-366
  SECTION("array handle is nullptr") {
    resc.res.array.array = nullptr;
    HIP_CHECK_ERROR(hipCreateSurfaceObject(&surf, &resc),
                    hipErrorInvalidHandle);
  }
#endif

#if HT_NVIDIA  // Disalbed due to defect EXSWHTEC-367
  SECTION("freed array handle") {
    hipArray_t invalid_array;
    HIP_CHECK(
        hipMallocArray(&invalid_array, &desc, 64, 0, hipArraySurfaceLoadStore));
    HIP_CHECK(hipFreeArray(invalid_array));
    resc.res.array.array = invalid_array;
    HIP_CHECK_ERROR(hipCreateSurfaceObject(&surf, &resc),
                    hipErrorContextIsDestroyed);
  }
#endif

  HIP_CHECK(hipFreeArray(array));
}

/**
 * Test Description
 * ------------------------
 *  - This test validates the basic functionality of hipCreateSurfaceObject
 *  - and hipDestorySurfaceObject apis during stream capture.
 * Test source
 * ------------------------
 *  - unit/surface/hipCreateSurfaceObject.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_StreamCapture_SurfaceObject_Create_Destroy") {
  CHECK_IMAGE_SUPPORT

  hipArray_t array = nullptr;
  size_t width = 64;
  size_t height = 64;
  hipChannelFormatDesc desc = hipCreateChannelDesc<int>();
  unsigned int flags = hipArraySurfaceLoadStore;

  HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));

  hipError_t error_capture = hipSuccess;
  BEGIN_CAPTURE_SYNC(error_capture, true);

  hipResourceDesc pResDesc;
  pResDesc.resType = hipResourceTypeArray;
  pResDesc.res.array.array = array;
  hipSurfaceObject_t pSurfObject;
#if HT_AMD  // SWDEV-510271 it will be removed once the defect is fixed.
  error_capture = hipSuccess;
#endif
  HIP_CHECK_ERROR(hipCreateSurfaceObject(&pSurfObject, &pResDesc),
                  error_capture);
  HIP_CHECK_ERROR(hipDestroySurfaceObject(pSurfObject), error_capture);

  END_CAPTURE_SYNC(error_capture);
  HIP_CHECK(hipFreeArray(array));
}

/**
 * End doxygen group SurfaceTest.
 * @}
 */
