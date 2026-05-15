/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "OCLTestListImp.h"

//
// Includes for tests
//
#ifdef _WIN32
#include "OCLDX11YUY2.h"
#endif

//
//  Helper macro for adding tests
//
template <typename T> static void* dictionary_CreateTestFunc(void) { return new T(); }

#define TEST(name) {#name, &dictionary_CreateTestFunc<name>}

#ifdef _WIN32

TestEntry TestList[] = {TEST(OCLDX11YUY2)};

unsigned int TestListCount = sizeof(TestList) / sizeof(TestList[0]);
#else
TestEntry TestList[] = {{"void", 0}};
unsigned int TestListCount = 0;

#endif
unsigned int TestLibVersion = 0;
const char* TestLibName = "ocldx";
