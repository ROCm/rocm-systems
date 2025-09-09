/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

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

#include "user_object_common.hh"

/**
 * Negative Test for API - hipUserObjectRelease
 1) Pass User Object as nullptr
 2) Pass initialRefcount as 0
 3) Pass initialRefcount as INT_MAX
 */
TEST_CASE("Unit_hipUserObjectRelease_Negative") {
  int* object = new int();
  REQUIRE(object != nullptr);

  hipUserObject_t hObject;
  HIP_CHECK(hipUserObjectCreate(&hObject, object, destroyIntObj, 1, hipUserObjectNoDestructorSync));
  REQUIRE(hObject != nullptr);

  SECTION("Pass User Object as nullptr") {
    HIP_CHECK_ERROR(hipUserObjectRelease(nullptr, 1), hipErrorInvalidValue);
  }
  SECTION("Pass initialRefcount as 0") {
    HIP_CHECK_ERROR(hipUserObjectRelease(hObject, 0), hipErrorInvalidValue);
  }
  SECTION("Pass initialRefcount as INT_MAX") { HIP_CHECK(hipUserObjectRelease(hObject, INT_MAX)); }
}