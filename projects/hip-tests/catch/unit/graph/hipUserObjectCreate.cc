/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#include "user_object_common.hh"

/*
  Functional Test for API - hipUserObjectCreate
1) Call hipUserObjectCreate once and release it by calling hipUserObjectRelease
2) Call hipUserObjectCreate refCount as X and release it by calling
   hipUserObjectRelease with same refCount.
3) Call hipUserObjectCreate, retain it by calling hipUserObjectRetain
   and release it by calling hipUserObjectRelease twice.
4) Call hipUserObjectCreate with refCount as X, retain it by calling
   hipUserObjectRetain with count as Y and release it by calling
   hipUserObjectRelease with count as X+Y.
*/

/* 1) Call hipUserObjectCreate once and release it by
      calling hipUserObjectRelease */
static void hipUserObjectCreate_Functional_1(void* object, void destroyObj(void*)) {
  hipUserObject_t hObject;
  HIP_CHECK(hipUserObjectCreate(&hObject, object, destroyObj, 1, hipUserObjectNoDestructorSync));
  REQUIRE(hObject != nullptr);
  HIP_CHECK(hipUserObjectRelease(hObject));
}

HIP_TEST_CASE(Unit_hipUserObjectCreate_Functional_1) {
  SECTION("Called with int Object") {
    int* object = new int();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_1(object, destroyIntObj);
  }
  SECTION("Called with float Object") {
    float* object = new float();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_1(object, destroyFloatObj);
  }
  SECTION("Called with Class Object") {
    BoxClass* object = new BoxClass();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_1(object, destroyClassObj);
  }
  SECTION("Called with Struct Object") {
    BoxStruct* object = new BoxStruct();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_1(object, destroyStructObj);
  }
}

/* 2) Call hipUserObjectCreate refCount as X and release it by
      calling hipUserObjectRelease with same refCount. */
static void hipUserObjectCreate_Functional_2(void* object, void destroyObj(void*)) {
  int refCount = 5;
  hipUserObject_t hObject;
  HIP_CHECK(
      hipUserObjectCreate(&hObject, object, destroyObj, refCount, hipUserObjectNoDestructorSync));
  REQUIRE(hObject != nullptr);
  HIP_CHECK(hipUserObjectRelease(hObject, refCount));
}

HIP_TEST_CASE(Unit_hipUserObjectCreate_Functional_2) {
  SECTION("Called with int Object") {
    int* object = new int();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_2(object, destroyIntObj);
  }
  SECTION("Called with float Object") {
    float* object = new float();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_2(object, destroyFloatObj);
  }
  SECTION("Called with Class Object") {
    BoxClass* object = new BoxClass();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_2(object, destroyClassObj);
  }
  SECTION("Called with Struct Object") {
    BoxStruct* object = new BoxStruct();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_2(object, destroyStructObj);
  }
}

/* 3) Call hipUserObjectCreate, retain it by calling hipUserObjectRetain
      and release it by calling hipUserObjectRelease twice. */
static void hipUserObjectCreate_Functional_3(void* object, void destroyObj(void*)) {
  hipUserObject_t hObject;
  HIP_CHECK(hipUserObjectCreate(&hObject, object, destroyObj, 1, hipUserObjectNoDestructorSync));
  REQUIRE(hObject != nullptr);
  HIP_CHECK(hipUserObjectRetain(hObject));
  HIP_CHECK(hipUserObjectRelease(hObject));
  HIP_CHECK(hipUserObjectRelease(hObject));
}

HIP_TEST_CASE(Unit_hipUserObjectCreate_Functional_3) {
  SECTION("Called with int Object") {
    int* object = new int();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_3(object, destroyIntObj);
  }
  SECTION("Called with float Object") {
    float* object = new float();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_3(object, destroyFloatObj);
  }
  SECTION("Called with Class Object") {
    BoxClass* object = new BoxClass();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_3(object, destroyClassObj);
  }
  SECTION("Called with Struct Object") {
    BoxStruct* object = new BoxStruct();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_3(object, destroyStructObj);
  }
}

/* 4) Call hipUserObjectCreate with refCount as X, retain it by calling
      hipUserObjectRetain with count as Y and release it by calling
      hipUserObjectRelease with count as X+Y. */
static void hipUserObjectCreate_Functional_4(void* object, void destroyObj(void*)) {
  int refCount = 5;
  int refCountRetain = 8;
  hipUserObject_t hObject;
  HIP_CHECK(
      hipUserObjectCreate(&hObject, object, destroyObj, refCount, hipUserObjectNoDestructorSync));
  REQUIRE(hObject != nullptr);
  HIP_CHECK(hipUserObjectRetain(hObject, refCountRetain));
  HIP_CHECK(hipUserObjectRelease(hObject, refCount + refCountRetain));
}

HIP_TEST_CASE(Unit_hipUserObjectCreate_Functional_4) {
  SECTION("Called with int Object") {
    int* object = new int();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_4(object, destroyIntObj);
  }
  SECTION("Called with float Object") {
    float* object = new float();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_4(object, destroyFloatObj);
  }
  SECTION("Called with Class Object") {
    BoxClass* object = new BoxClass();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_4(object, destroyClassObj);
  }
  SECTION("Called with Struct Object") {
    BoxStruct* object = new BoxStruct();
    REQUIRE(object != nullptr);
    hipUserObjectCreate_Functional_4(object, destroyStructObj);
  }
}

/**
 * Negative Test for API - hipUserObjectCreate
 1) Pass User Object as nullptr
 2) Pass object as nullptr
 3) Pass Callback function as nullptr
 4) Pass initialRefcount as 0
 5) Pass initialRefcount as INT_MAX
 6) Pass flag other than hipUserObjectNoDestructorSync
 */
HIP_TEST_CASE(Unit_hipUserObjectCreate_Negative) {
  int* object = new int();
  REQUIRE(object != nullptr);

  hipUserObject_t hObject;
  SECTION("Pass User Object as nullptr") {
    HIP_CHECK_ERROR(
        hipUserObjectCreate(nullptr, object, destroyIntObj, 1, hipUserObjectNoDestructorSync),
        hipErrorInvalidValue);
    delete object;
  }
  SECTION("Pass object as nullptr") {
    HIP_CHECK(
        hipUserObjectCreate(&hObject, nullptr, destroyIntObj, 1, hipUserObjectNoDestructorSync));
    HIP_CHECK(hipUserObjectRelease(hObject, 1));
    delete object;
  }
  SECTION("Pass Callback function as nullptr") {
    HIP_CHECK_ERROR(
        hipUserObjectCreate(&hObject, object, nullptr, 1, hipUserObjectNoDestructorSync),
        hipErrorInvalidValue);
    delete object;
  }
  SECTION("Pass initialRefcount as 0") {
    HIP_CHECK_ERROR(
        hipUserObjectCreate(&hObject, object, destroyIntObj, 0, hipUserObjectNoDestructorSync),
        hipErrorInvalidValue);
    delete object;
  }
  SECTION("Pass initialRefcount as INT_MAX") {
    HIP_CHECK(hipUserObjectCreate(&hObject, object, destroyIntObj, INT_MAX,
                                  hipUserObjectNoDestructorSync));
    HIP_CHECK(hipUserObjectRelease(hObject, INT_MAX));
  }
  SECTION("Pass flag other than hipUserObjectNoDestructorSync") {
    HIP_CHECK_ERROR(hipUserObjectCreate(&hObject, object, destroyIntObj, 1, hipUserObjectFlags(9)),
                    hipErrorInvalidValue);
    delete object;
  }
}

HIP_TEST_CASE(Unit_hipUserObj_Negative_Test) {
  int* object = new int();
  REQUIRE(object != nullptr);

  hipUserObject_t hObject;

  // Create a new hObject with 2 reference
  HIP_CHECK(hipUserObjectCreate(&hObject, object, destroyIntObj, 2, hipUserObjectNoDestructorSync));
  REQUIRE(hObject != nullptr);

  // Release more than created. Will not release the references.
  HIP_CHECK(hipUserObjectRelease(hObject, 4));

  // Retain reference
  HIP_CHECK(hipUserObjectRetain(hObject, 1));

  // Release all the references pending
  HIP_CHECK(hipUserObjectRelease(hObject, 3));
}
