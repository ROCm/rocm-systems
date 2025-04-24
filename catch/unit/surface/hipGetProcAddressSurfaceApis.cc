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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <hip_test_helper.hh>
#include <utils.hh>

 /**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of hipCreateSurfaceObject API,
 *  - hipDestroySurfaceObject API from the hipGetProcAddress API
 *  - and then validates the basic functionality of those APIs
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/graph/hipGetProcAddress_Surface_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_Surface") {
  CHECK_IMAGE_SUPPORT

  void* hipCreateSurfaceObject_ptr = nullptr;
  void* hipDestroySurfaceObject_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "hipCreateSurfaceObject",
            &hipCreateSurfaceObject_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "hipDestroySurfaceObject",
            &hipDestroySurfaceObject_ptr,
            currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipCreateSurfaceObject_ptr)(hipSurfaceObject_t *,
    const hipResourceDesc *) =
    reinterpret_cast<hipError_t (*)(hipSurfaceObject_t *,
    const hipResourceDesc *)>
    (hipCreateSurfaceObject_ptr);
  hipError_t (*dyn_hipDestroySurfaceObject_ptr)(hipSurfaceObject_t) =
    reinterpret_cast<hipError_t (*)(hipSurfaceObject_t)>
    (hipDestroySurfaceObject_ptr);

  size_t width = 64;
  size_t height = 64;

  hipArray_t array = nullptr;
  hipChannelFormatDesc desc = hipCreateChannelDesc<int>();
  unsigned int flags = hipArraySurfaceLoadStore;
  HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
  REQUIRE(array != nullptr);

  hipResourceDesc pResDesc;
  pResDesc.resType = hipResourceTypeArray;
  pResDesc.res.array.array = array;

  hipSurfaceObject_t pSurfObject = nullptr;
  HIP_CHECK(dyn_hipCreateSurfaceObject_ptr(&pSurfObject, &pResDesc));
  REQUIRE(pSurfObject != nullptr);

  HIP_CHECK(dyn_hipDestroySurfaceObject_ptr(pSurfObject));
  HIP_CHECK(hipFreeArray(array));
}
