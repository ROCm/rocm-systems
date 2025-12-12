/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#include "user_object_common.hh"

/**
 * Negative Test for API - hipUserObjectRetain
 1) Pass User Object as nullptr
 2) Pass initialRefcount as 0
 3) Pass initialRefcount as INT_MAX
 */
HIP_TEST_CASE(Unit_hipUserObjectRetain_Negative) {
  int* object = new int();
  REQUIRE(object != nullptr);

  hipUserObject_t hObject;
  HIP_CHECK(hipUserObjectCreate(&hObject, object, destroyIntObj, 1, hipUserObjectNoDestructorSync));
  REQUIRE(hObject != nullptr);

  SECTION("Pass User Object as nullptr") {
    HIP_CHECK_ERROR(hipUserObjectRetain(nullptr, 1), hipErrorInvalidValue);
  }
  SECTION("Pass initialRefcount as 0") {
    HIP_CHECK_ERROR(hipUserObjectRetain(hObject, 0), hipErrorInvalidValue);
  }
  SECTION("Pass initialRefcount as INT_MAX") {
    HIP_CHECK(hipUserObjectRetain(hObject, INT_MAX));
    HIP_CHECK(hipUserObjectRelease(hObject, INT_MAX));
  }
  HIP_CHECK(hipUserObjectRelease(hObject, 1));
}
