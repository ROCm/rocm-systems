/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

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
