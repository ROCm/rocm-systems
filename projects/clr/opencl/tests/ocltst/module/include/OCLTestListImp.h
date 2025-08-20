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

#ifndef __Dictionary_h__
#define __Dictionary_h__

//
// Testing module (plugin) interface forward declarations
//
#ifdef _WIN32
#define OCL_DLLEXPORT __declspec(dllexport)
#define OCL_CALLCONV __cdecl
#endif
#ifdef __linux__
#define OCL_DLLEXPORT
#define OCL_CALLCONV
#endif

class OCLTest;

//
//  OCLTestList_TestCount - retrieve the number of tests in the testing module
//
extern "C" OCL_DLLEXPORT unsigned int OCL_CALLCONV OCLTestList_TestCount(void);

//
//  OCLTestList_TestLibVersion - retrieve the version of test lib in the testing
//  module
//
extern "C" OCL_DLLEXPORT unsigned int OCL_CALLCONV OCLTestList_TestLibVersion(void);

//
//  OCLTestList_TestLibName - retrieve the name of test library
//
extern "C" OCL_DLLEXPORT const char* OCL_CALLCONV OCLTestList_TestLibName(void);

//
//  OCLTestList_TestName - retrieve the name of the indexed test in the module
//
extern "C" OCL_DLLEXPORT const char* OCL_CALLCONV OCLTestList_TestName(unsigned int testNum);

//
//  OCLTestList_CreateTest - create a test by index
//
extern "C" OCL_DLLEXPORT OCLTest* OCL_CALLCONV OCLTestList_CreateTest(unsigned int testNum);

//
//  OCLTestList_DestroyTest - destroy a test object
//
extern "C" OCL_DLLEXPORT void OCL_CALLCONV OCLTestList_DestroyTest(OCLTest* test);

//
//  internal global data that is populated in each dll
//
typedef struct _TestEntry {
  const char* name;
  void* (*create)(void);
} TestEntry;

extern TestEntry TestList[];
extern unsigned int TestListCount;
extern unsigned int TestLibVersion;
extern const char* TestLibName;

#endif
