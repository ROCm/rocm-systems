/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "OCLTestListImp.h"

#include <stdlib.h>

#include "OCLTest.h"

//
//  OCLTestList_TestCount - retrieve the number of tests in the testing module
//
unsigned int OCL_CALLCONV OCLTestList_TestCount(void) { return TestListCount; }

//
//  OCLTestList_TestLibVersion - retrieve the version of test lib in the testing
//  module
//
unsigned int OCL_CALLCONV OCLTestList_TestLibVersion(void) { return TestLibVersion; }

//
//  OCLTestList_TestLibName - retrieve the name of test library
//
const char* OCL_CALLCONV OCLTestList_TestLibName(void) { return TestLibName; }

//
//  OCLTestList_TestName - retrieve the name of the indexed test in the module
//
const char* OCL_CALLCONV OCLTestList_TestName(unsigned int testNum) {
  if (testNum >= OCLTestList_TestCount()) {
    return NULL;
  }

  return TestList[testNum].name;
}

//
//  OCLTestList_CreateTest - create a test by index
//
OCLTest* OCL_CALLCONV OCLTestList_CreateTest(unsigned int testNum) {
  if (testNum >= OCLTestList_TestCount()) {
    return NULL;
  }

  return reinterpret_cast<OCLTest*>((*TestList[testNum].create)());
}

//
//  OCLTestList_DestroyTest - destroy a test object
//
void OCL_CALLCONV OCLTestList_DestroyTest(OCLTest* test) { delete test; }
