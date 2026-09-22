/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

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
