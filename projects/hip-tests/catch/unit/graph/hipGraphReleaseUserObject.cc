/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#include "user_object_common.hh"

/**
 * Negative Test for API - hipGraphReleaseUserObject
 1) Pass graph as nullptr
 2) Pass User Object as nullptr
 3) Pass initialRefcount as 0
 4) Pass initialRefcount as INT_MAX
 */
TEST_CASE(Unit_hipGraphReleaseUserObject_Negative) {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  float* object = new float();
  REQUIRE(object != nullptr);
  hipUserObject_t hObject;

  HIP_CHECK(
      hipUserObjectCreate(&hObject, object, destroyFloatObj, 1, hipUserObjectNoDestructorSync));
  REQUIRE(hObject != nullptr);
  HIP_CHECK(hipGraphRetainUserObject(graph, hObject, 1, hipGraphUserObjectMove));

  SECTION("Pass graph as nullptr") {
    HIP_CHECK_ERROR(hipGraphReleaseUserObject(nullptr, hObject, 1), hipErrorInvalidValue);
  }
  SECTION("Pass User Object as nullptr") {
    HIP_CHECK_ERROR(hipGraphReleaseUserObject(graph, nullptr, 1), hipErrorInvalidValue);
  }
  SECTION("Pass initialRefcount as 0") {
    HIP_CHECK_ERROR(hipGraphReleaseUserObject(graph, hObject, 0), hipErrorInvalidValue);
  }
  SECTION("Pass initialRefcount as INT_MAX") {
    HIP_CHECK(hipGraphReleaseUserObject(graph, hObject, INT_MAX));
  }

  HIP_CHECK(hipUserObjectRelease(hObject, 1));
  HIP_CHECK(hipGraphDestroy(graph));
}